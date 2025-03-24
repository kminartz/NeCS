import warnings

import torch
import torch.nn as nn
from modules.engine.GNN_modules import MessagePassingGNN
EPS = 1e-2

class Encoder(nn.Module):

    def __init__(self, emb_dim=16):
        super().__init__()
        self.arch = nn.Sequential(nn.LazyLinear(emb_dim))


    def forward(self, x):
        """
        x can be tuple of (nodes, *some other stuff) or just nodes. IN case of the tuple, only the nodes are embedded
        and the other stuff is passed through unchanged
        """
        if isinstance(x, tuple):
            # x can be a tuple, because we could also pass edge feats or edge index
            return self.arch(x[0]), *x[1:]
        return self.arch(x)


class Probabilistic_Decoder(nn.Module):
    """
    This is a conditional vae that takes the embedded output produced by the 'Processor' class and produces a final
    embedding that can be mapped to the data space.
    """

    def __init__(self, num_layers: int, emb_dim: int, dynamic_channels: tuple,
                 num_latent_dims: int, encoder_net: nn.Module, prior_net: nn.Module, end_arch:nn.Module,
                 free_bits: float,
                 ):

        """
        :param num_layers: number of layers in the decoder
        :param emb_dim: hidden dimensionality of the model
        :param dynamic_channels: channels that are predicted by the model
        :param num_latent_dims: number of latent dimensions
        :param encoder_net: encoder network that maps the input to the latent space
        :param prior_net: prior network that maps the input to the latent space
        :param end_arch: architecture that maps the latent space to an output embedding which can be mapped to data space
        in the wrapper classes
        :param free_bits: number of bits that are not penalized in the KL loss
        :param embedding_decoder_dropout_prob: dropout probability for the embedding decoder

        """


        super().__init__()
        self.num_layers = num_layers
        self.num_latent_dims = num_latent_dims
        self.encoder_net = encoder_net
        self.prior_net = prior_net
        self.end_arch = end_arch

        self.dynamic_channels = dynamic_channels

        self.emb_dim = emb_dim

        self.enc_mu = nn.LazyLinear(num_latent_dims)
        self.enc_log_sigma = nn.LazyLinear(num_latent_dims)

        self.prior_mu = nn.LazyLinear(num_latent_dims)
        self.prior_log_sigma = nn.LazyLinear(num_latent_dims)
        self.free_bits = free_bits

        # add an embedder that embeds x_true to a larger embedding space to we can increase its importance more easily
        self.x_true_embedder = nn.Sequential(nn.LazyLinear(self.emb_dim), nn.ReLU(), nn.LazyLinear(self.emb_dim))


    def forward(self, tup_of_e_and_x_true):
        """
        :param tup_of_e_and_x_true: tuple of (e, x_true) where e is the output of the processor and x_true is the
        ground-truth at time t+1. e and x_true itself are also tuples of format (nodes, edge_idx, edge_feat)
        """

        e, x_true = tup_of_e_and_x_true
        assert isinstance(e, tuple), 'e should be tuple of (nodes, edge_idx, edge_feat)'
        assert (x_true is None) or isinstance(x_true, tuple), 'x_true should be tuple of (nodes, edge_idx, edge_feat)'

        # get prior distribution, only conditioned on current state:
        mu_prior, log_sigma_prior = self.vae_prior(e)
        sigma_prior = (nn.Softplus()(log_sigma_prior) + EPS)
        prior_dist = torch.distributions.Normal(mu_prior, sigma_prior)
        kl_loss = torch.tensor(0)

        # do posterior sampling when x_true is available:
        if x_true is not None:
            # we are in training mode and have the ground-truth available -> sample from approx. posterior
            mu_enc, log_sigma_enc = self.vae_encoder(e, x_true)  # shape (bs, num_latent_dim)
            enc_dist = torch.distributions.Normal(mu_enc, nn.Softplus()(log_sigma_enc) + EPS)
            kl_loss = self.kl_loss(enc_dist, prior_dist)
        else:
            # we are in eval mode and don't have the approx posterior -> sample from prior
            enc_dist = prior_dist

        z = self.reparameterization(enc_dist)  # (bs, latent_dim)

        bs = z.shape[0]

        # apply the decoder:

        node_emb = e[0]  # discard edges
        final_emb = torch.cat([node_emb, z], dim=2)
        final_input = (final_emb, *e[1:])  # concat edges again
        out = self.end_arch(final_input)

        # apply the free bits objective: (See Kingma IAF paper appendix)
        kl_avgd_over_batch = torch.sum(kl_loss, dim=0, keepdim=True) / bs
        low_kl_idx = kl_avgd_over_batch < self.free_bits

        # Don't penalize the KL loss for dimensions where the avg KL for that dim is already very low
        # (implemented by detaching from the compute graph at those dimensions)
        kl_loss_with_detach = torch.where(low_kl_idx, kl_loss.detach(), kl_loss)
        additional_loss = torch.sum(kl_loss_with_detach)

        return out, additional_loss

    def vae_encoder(self, x, x_true):

        assert isinstance(x, tuple), 'x should be tuple of (nodes, edge_idx, edge_feat)'
        assert isinstance(x_true, tuple), 'x_true should be tuple of (nodes, edge_idx, edge_feat)'
        assert x[2].shape[0] == x_true[2].shape[0] or x_true[2].shape[0] == 0, \
            'Only support same topology at t and t+1 for the encoder, or no edge feat for t+1!'

        x = (torch.cat([x[0], self.x_true_embedder(x_true[0])], dim=-1),  # node feat
             x[1], torch.cat([x[2], x_true[2]], dim=-1) if x_true[2].shape[0] > 0 else x[2],  # edge attr,
             *x[3:]  # type and misc info
             )  # concat node features

        enc = self.encoder_net(x)  # (bs, latent)
        mu = self.enc_mu(enc)
        log_sigma = self.enc_log_sigma(enc)
        return mu, log_sigma

    def vae_prior(self, x):

        p = self.prior_net(x)
        mu = self.prior_mu(p)
        log_sigma = self.prior_log_sigma(p)
        return mu, log_sigma

    def reparameterization(self, dist: torch.distributions.Distribution) -> torch.Tensor:
        z = dist.rsample()
        # z = dist.mean + torch.randn_like(dist.mean) * (dist.scale) * 1.5
        return z

    def kl_loss(self, enc_dist: torch.distributions.Distribution, prior_dist: torch.distributions.Distribution):
        return torch.distributions.kl_divergence(enc_dist, prior_dist)




class Processor(nn.Module):

    def __init__(self, num_layers, emb_dim, nonlinearity=nn.ReLU(),
                 channels_cat_to_output=None, **kwargs):

        """
        :param num_layers: number of layers in the processor
        :param emb_dim: hidden dimensionality of the model
        :param nonlinearity: nonlinearity to use in the processor
        :param processor_type_str: string indicating the type of processor to use. For the pedestrian codebase, so
        far only 'flexiblegnn' is implemended, but other options could be added
        :param channels_cat_to_output: if not None, the channels specified by this tuple are concatenated to the output of the processor
        :param kwargs: additional keyword arguments passed to the processor architecture

        """

        super().__init__()

        self.channels_cat_to_output = channels_cat_to_output

        modules = []
        modules.append(MessagePassingGNN(num_layers, emb_dim, nonlinearity, **kwargs))
        self.arch = nn.Sequential(*modules)

    def forward(self, e, x):
        """
        :param e: output of the encoder
        :param x: input data
        """
        out = self.arch(e)  # tuple of (nodes, edge_idx, edge_feat), where edge feat are the updated edge embeddings

        if self.channels_cat_to_output is not None:  # static conditioning signal, concatenate this to the output
            out_nodes_w_static = torch.cat([x[0][..., self.channels_cat_to_output], out[0]], dim=2)
        out = (out_nodes_w_static, *out[1:])  # again add the edge features (and possibly x and v, edge_index, batch_index ...) for input to the decoder

        return out


class Enc_Proc_Dec(nn.Module):
    """
    main class that does the fw pass of the encoder, processor and decoder -- produces an output embedding that
    can be mapped to the data space by the wrapper class
    """

    def __init__(self, num_layers, im_dim=2, emb_dim=16, nonlinearity=nn.ReLU(),
                 dynamic_channels=None,
                 num_latent_dim=0, free_bits=0,
                 num_prior_and_encoder_layers=None,
                 **kwargs):

        """
        :param num_layers: number of layers in the processor and decoder decoding architecture (which processes z and the processor's output)
        :param im_dim: total number of channels/features in the input
        :param emb_dim: hidden dimensionality of the model
        :param nonlinearity: nonlinearity to use
        :param dynamic_channels: channels that are predicted by the model
        :param num_latent_dim: number of latent dimensions
        :param free_bits: number of bits that are not penalized in the KL loss (dimension-wise)
        :param prior_and_encoder_arch: string indicating the architecture of the prior and encoder networks. For the pedestrian codebase,
        only 'flexiblegnn' is implemented, but more can be added
        :param num_prior_and_encoder_layers: number of layers in the prior and encoder networks. If not specified, will use 'num_layers'
        :param do_node_aggregation: if True, the node embeddings are aggregated before being mapped to the latent space,
        which will lead to a permutation-invariant distribution.
        :param embedding_decoder_dropout_prob: dropout probability for the processor output before it is passed to the decoder
        :param kwargs: additional keyword arguments passed to the architectures
        """

        super().__init__()

        #######
        # store some general information:
        #######

        if dynamic_channels is None:  # all channels are assumed to be predicted
            self.dynamic_channels = (i for i in range(im_dim))
        else:
            self.dynamic_channels = dynamic_channels  # set the channels that actually change over time

        all_channels = set([i for i in range(im_dim)])
        self.static_channels = tuple(all_channels - set(self.dynamic_channels)) # channels that remain fixed over time
        self.num_latent_dim = num_latent_dim
        if num_prior_and_encoder_layers is None:
            num_prior_and_encoder_layers = num_layers

        #####
        # PROCESSOR
        #####

        self.proc = Processor(num_layers, emb_dim, nonlinearity, self.static_channels,
                              **kwargs)

        #####
        # ENCODER
        #####

        self.enc = Encoder(emb_dim)  # downsampling factor only has effect on grids

        #####
        # DECODER
        #####

        vae_encoder_net, vae_prior_net = get_vae_enc_and_prior_nets(emb_dim, num_prior_and_encoder_layers,
                                                                    nonlinearity,
                                                                    **kwargs)
        decoder_end_arch = MessagePassingGNN(num_layers=num_layers, emb_dim=emb_dim, activation=nonlinearity, **kwargs)

        self.dec = Probabilistic_Decoder(num_layers, emb_dim, dynamic_channels, num_latent_dims=num_latent_dim,
                                             encoder_net=vae_encoder_net,  prior_net=vae_prior_net,
                                             end_arch=decoder_end_arch, free_bits=free_bits,)

    def forward(self, x, x_true=None):
        """
        :param x: input data -- tuple of (nodes, edge_idx, edge_feat)
        :param x_true: ground-truth data at time t+1 -- tuple of (nodes, edge_idx, edge_feat)
        """
        assert isinstance(x, tuple), 'input data should be a tuple of (nodes, edge_idx, edge_feat)'
        assert x_true is None or isinstance(x_true, tuple), 'ground-truth data should be a tuple of (nodes, edge_idx, edge_feat)'
        # get the ground-truth at time t+1, if available:

        x_true_dynamic = (x_true[0][..., self.dynamic_channels], *x_true[1:]) if x_true is not None else None #index_channels_for_domain(x_true, self.dynamic_channels, 'graph') if x_true is not None else None

        # initialize additional loss terms at 0:
        additional_loss = torch.sum(torch.zeros(size=(1,)))

        # encode:
        e = self.enc(x)

        # process:
        input_to_decoder = self.proc(e, x)

        # decode:
        d, add_loss = self.dec((input_to_decoder, x_true_dynamic))
        additional_loss = additional_loss + add_loss

        return d, additional_loss



def get_vae_enc_and_prior_nets(emb_dim, num_layers, nonlinearity=nn.ReLU(), **kwargs):
    # method for getting the required prior and encoder nets

    encoder_net = nn.Sequential(MessagePassingGNN(num_layers, emb_dim, nonlinearity, **kwargs),
                                Lambda(lambda x: x[0])  # keep only nodes
                                )
    prior_net = nn.Sequential(MessagePassingGNN(num_layers, emb_dim, nonlinearity, **kwargs),
                              Lambda(lambda x: x[0])  # keep only nodes
                              )
    return encoder_net, prior_net



class Lambda(nn.Module):

    def __init__(self, lambd: callable):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

