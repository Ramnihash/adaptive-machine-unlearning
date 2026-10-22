from dataclasses import dataclass

import torch

from data.preprocessing import ProcessedData, preprocess_movielens


@dataclass
class InteractionGraph:
    edge_index: torch.Tensor
    num_users: int
    num_movies: int
    num_nodes: int
    num_edges: int


def build_interaction_graph(
    data: ProcessedData | None = None,
) -> InteractionGraph:
    if data is None:
        data = preprocess_movielens()

    user_indices = torch.tensor(
        data.ratings["user_idx"].to_numpy(),
        dtype=torch.long,
    )
    movie_indices = torch.tensor(
        data.ratings["movie_idx"].to_numpy(),
        dtype=torch.long,
    )

    movie_nodes = movie_indices + data.num_users

    source_nodes = torch.cat((user_indices, movie_nodes))
    target_nodes = torch.cat((movie_nodes, user_indices))

    edge_index = torch.stack((source_nodes, target_nodes), dim=0)

    num_nodes = data.num_users + data.num_movies

    return InteractionGraph(
        edge_index=edge_index,
        num_users=data.num_users,
        num_movies=data.num_movies,
        num_nodes=num_nodes,
        num_edges=edge_index.shape[1],
    )