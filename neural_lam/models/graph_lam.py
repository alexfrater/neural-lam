# Third-party
import torch_geometric as pyg

# First-party
from neural_lam import utils
from neural_lam.interaction_net import InteractionNet
from neural_lam.models.base_graph_model import BaseGraphModel
from neural_lam.utils import ExpandToBatch

import torch
class GraphLAM(BaseGraphModel):
    """
    Full graph-based LAM model that can be used with different
    (non-hierarchical )graphs. Mainly based on GraphCast, but the model from
    Keisler (2022) is almost identical. Used for GC-LAM and L1-LAM in
    Oskarsson et al. (2023).
    """

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        assert (
            not self.hierarchical
        ), "GraphLAM does not use a hierarchical mesh graph"

        # grid_dim from data + static + batch_static
        mesh_dim = self.mesh_static_features.shape[1]
        m2m_edges, m2m_dim = self.m2m_features.shape
        print(
            f"Edges in subgraphs: m2m={m2m_edges}, g2m={self.g2m_edges}, "
            f"m2g={self.m2g_edges}"
        )

        # Define sub-models
        # Feature embedders for mesh
        self.mesh_embedder = utils.make_mlp([mesh_dim] + self.mlp_blueprint_end)
        self.m2m_embedder = utils.make_mlp([m2m_dim] + self.mlp_blueprint_end)

        # GNNs
        # processor

        # Create a single instance of InteractionNet
        processor_net = InteractionNet(
            self.m2m_edge_index,
            args.hidden_dim,
            hidden_layers=args.hidden_layers,
            aggr=args.mesh_aggr,
        )
        self.processor = pyg.nn.Sequential(
            "mesh_rep, edge_rep",
            [(processor_net, "mesh_rep, mesh_rep, edge_rep -> mesh_rep, edge_rep")]
        )

        


    

        self.m2m_expander = ExpandToBatch()

        # processor_nets = [
        #     InteractionNet(
        #         self.m2m_edge_index,
        #         args.hidden_dim,
        #         hidden_layers=args.hidden_layers,
        #         aggr=args.mesh_aggr,
        #     )
        #     for _ in range(args.processor_layers)
        # ]
        # self.processor = pyg.nn.Sequential(
        #     "mesh_rep, edge_rep",
        #     [
        #         (net, "mesh_rep, mesh_rep, edge_rep -> mesh_rep, edge_rep")
        #         for net in processor_nets
        #     ],
        # )


        # self.processor = pyg.nn.Sequential("mesh_rep, edge_rep", (processor_nets, "mesh_rep, mesh_rep, edge_rep -> mesh_rep, edge_rep")
            
        # )


    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        return self.mesh_static_features.shape[0], 0

    def embedd_mesh_nodes(self):
        """
        Embed static mesh features
        Returns tensor of shape (N_mesh, d_h)
        """
        return self.mesh_embedder(self.mesh_static_features)  # (N_mesh, d_h)


    def preprocess_inputs(self, batch):
        # batch = None
        # for data_batch in dataloader:
        #     batch = batch  # Trigger forward pass
        #     break
        # print("batch", batch)
        (
            init_states,
            target_states,
            batch_static_features,
            forcing_features,
        ) = batch
        prev_state = init_states[:, 1]
        prev_prev_state = init_states[:, 1]
        forcing = forcing_features[:, 0]


        # Create full grid node features of shape (B, num_grid_nodes, grid_dim)
        grid_features = torch.cat(
            (
                prev_state,
                prev_prev_state,
                batch_static_features,
                forcing,
                self.expand_to_batch(self.grid_static_features, 1),
                # self.grid_static_features,
            ),
            dim=-1,
        )
        #Not using batch sizes greater than 1 for ample
        grid_features = grid_features.squeeze(0)
        def filter_edges(edge_index, num_nodes_to_keep):
            # Create a mask that keeps only edges between nodes within the first 50 nodes
            mask = (edge_index[0, :] < num_nodes_to_keep) & (edge_index[1, :] < num_nodes_to_keep)
    
            # Filter the edge index tensor using the mask
            edge_index_filtered = edge_index[:, mask]
            
            return edge_index_filtered

        print('m2g features',self.m2g_features)

        if self.args.subset_ds:
            num_nodes_to_keep = 50

            self.m2m_features = self.m2m_features[:num_nodes_to_keep, :]
            self.mesh_static_features = self.mesh_static_features[:num_nodes_to_keep, :]
            self.m2g_features = self.m2g_features[:num_nodes_to_keep, :]
            self.g2m_features = self.g2m_features[:num_nodes_to_keep, :]
            grid_features = grid_features[:num_nodes_to_keep, :]
        
            self.g2m_edge_index = filter_edges(self.g2m_edge_index, num_nodes_to_keep)
            self.m2g_edge_index = filter_edges(self.m2g_edge_index, num_nodes_to_keep)

        print('M2M Features:', self.m2m_features.shape)
        print('Mesh Static Features:', self.mesh_static_features.shape)
        print('M2G Features:', self.m2g_features.shape)
        print('G2M Features:', self.g2m_features.shape)
        print('Grid Features:', grid_features.shape)
        print('G2M Edge Index:', self.g2m_edge_index.shape)
        print('M2G Edge Index:', self.m2g_edge_index.shape)
        
        return [
            [self.m2m_features,
            self.mesh_static_features,
            self.m2g_features,
            self.g2m_features,
            grid_features],
            [self.g2m_edge_index,
            self.m2g_edge_index]
            ]
        # {
        # "grid_features": grid_features,
        # "g2m_features": self.g2m_features,
        # "edge_index1": self.g2m_edge_index,
        # "node_features2": self.mesh_static_features,
        # "edge_features2": self.m2g_features,
        # "edge_index2": self.m2g_edge_index
        # }

    
    def process_step(self, mesh_rep):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, N_mesh, d_h)
        Returns mesh_rep: (B, N_mesh, d_h)
        """
        # Embed m2m here first
        batch_size = mesh_rep.shape[0]
        m2m_emb = self.m2m_embedder(self.m2m_features)  # (M_mesh, d_h)
        # m2m_emb = self.m2m_expander(
        #     m2m_emb, batch_size
        # )  # (B, M_mesh, d_h)
   
        # mesh_rep= mesh_rep.squeeze(0)

        mesh_rep, _ = self.processor(
            mesh_rep, m2m_emb
        )  # (B, N_mesh, d_h)
        return mesh_rep

