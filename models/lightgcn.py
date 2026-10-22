
import torch
from torch import nn

from graph.graph_builder import InteractionGraph


class LightGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_movies: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()

        if num_users <= 0 or num_movies <= 0:
            raise ValueError("Number of users and movies must be positive.")

        if embedding_dim <= 0 or num_layers < 0:
            raise ValueError("Invalid embedding dimension or layer count.")

        self.num_users = num_users
        self.num_movies = num_movies
        self.num_nodes = num_users + num_movies
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.num_nodes, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

    def normalize_graph(self, edge_index: torch.Tensor) -> torch.Tensor:
        source, target = edge_index

        degree = torch.bincount(
            source,
            minlength=self.num_nodes,
        ).to(dtype=self.embedding.weight.dtype)

        inverse_sqrt_degree = degree.clamp(min=1).pow(-0.5)

        edge_weights = (
            inverse_sqrt_degree[source]
            * inverse_sqrt_degree[target]
        )

        indices = torch.stack((source, target), dim=0)

        with torch.sparse.check_sparse_tensor_invariants():
            adjacency = torch.sparse_coo_tensor(
                indices,
                edge_weights,
                size=(self.num_nodes, self.num_nodes),
                device=edge_index.device,
            ).coalesce()
        return adjacency

    def propagate(self, adjacency: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding.weight
        all_embeddings = [embeddings]

        for _ in range(self.num_layers):
            embeddings = torch.sparse.mm(adjacency, embeddings)
            all_embeddings.append(embeddings)

        return torch.stack(all_embeddings, dim=0).mean(dim=0)

    def forward(
        self,
        user_indices: torch.Tensor,
        movie_indices: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        all_embeddings = self.propagate(adjacency)

        user_embeddings = all_embeddings[user_indices]
        movie_nodes = movie_indices + self.num_users
        movie_embeddings = all_embeddings[movie_nodes]

        return (user_embeddings * movie_embeddings).sum(dim=-1)

    def get_all_embeddings(
        self,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_embeddings = self.propagate(adjacency)

        user_embeddings = all_embeddings[:self.num_users]
        movie_embeddings = all_embeddings[
            self.num_users:self.num_nodes
        ]

        return user_embeddings, movie_embeddings


def create_lightgcn(
    graph: InteractionGraph,
    embedding_dim: int = 64,
    num_layers: int = 3,
) -> LightGCN:
    return LightGCN(
        num_users=graph.num_users,
        num_movies=graph.num_movies,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
    )
