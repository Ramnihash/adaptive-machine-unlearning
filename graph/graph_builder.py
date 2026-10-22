from dataclasses import dataclass

import pandas as pd
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

    return build_graph_from_ratings(
        data.ratings,
        num_users=data.num_users,
        num_movies=data.num_movies,
    )


def build_graph_from_ratings(
    ratings: pd.DataFrame,
    num_users: int,
    num_movies: int,
) -> InteractionGraph:
    """Build the bipartite graph from the given interactions only.

    Pass the training split here so validation/test edges never leak
    into message passing.
    """
    user_indices = torch.tensor(
        ratings["user_idx"].to_numpy(),
        dtype=torch.long,
    )
    movie_indices = torch.tensor(
        ratings["movie_idx"].to_numpy(),
        dtype=torch.long,
    )

    movie_nodes = movie_indices + num_users

    source_nodes = torch.cat((user_indices, movie_nodes))
    target_nodes = torch.cat((movie_nodes, user_indices))

    edge_index = torch.stack((source_nodes, target_nodes), dim=0)

    num_nodes = num_users + num_movies

    return InteractionGraph(
        edge_index=edge_index,
        num_users=num_users,
        num_movies=num_movies,
        num_nodes=num_nodes,
        num_edges=edge_index.shape[1],
    )
