from deeplake.core import vectorstore
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.core.vectorstore.vector_search import filter as filter_utils
from deeplake.core.vectorstore.vector_search import utils
from deeplake.core.dataset import Dataset as DeepLakeDataset
from typing import Union, Dict
import os
import time

EMBEDDINGS_CACHE = {}

import numpy as np

def fetch_embeddings_in_batches(view, embedding_tensor, batch_size=100*1000):
    # Fetch one sample to determine the embedding size
    sample_embedding = dataset_utils.fetch_embeddings(
        view=view[0:1],
        embedding_tensor=embedding_tensor,
    )
    embedding_size = sample_embedding.shape[1]
    num_items = len(view)
    
    # Create an empty matrix of float16 to save memory
    all_embeddings = np.empty((num_items, embedding_size), dtype=np.float16)
    
    for start_idx in range(0, num_items, batch_size):
        end_idx = min(start_idx + batch_size, num_items)
        # Load batch embeddings
        batch_embeddings = dataset_utils.fetch_embeddings(
            view=view[start_idx:end_idx],
            embedding_tensor=embedding_tensor,
        )
        # Convert to fp16 and store in the preallocated matrix
        all_embeddings[start_idx:end_idx] = batch_embeddings.astype(np.float16)
    
    return all_embeddings


def vector_search(
    query,
    query_emb,
    exec_option,
    dataset,
    logger,
    filter,
    embedding_tensor,
    distance_metric,
    k,
    return_tensors,
    return_view,
    token,
    org_id,
    return_tql,
) -> Union[Dict, DeepLakeDataset]:
    
    recursive_retrieval = os.getenv('recursive_retrieval', 'True').lower() in ['false', '0']
    
    if query is not None:
        raise NotImplementedError(
            f"User-specified TQL queries are not supported for exec_option={exec_option} "
        )

    if return_tql:
        raise NotImplementedError(
            f"return_tql is not supported for exec_option={exec_option}"
        )

    view = filter_utils.attribute_based_filtering_python(dataset, filter)

    # Use a unique key for caching. Here, it's simple but can be extended to be more sophisticated
    cache_key = "no_filter" if filter is None else str(filter)

    return_data = {}

    # Only fetch embeddings and run the search algorithm if an embedding query is specified
    if query_emb is not None:
        start_time = time.time()
        if recursive_retrieval and cache_key in EMBEDDINGS_CACHE:
            embeddings = EMBEDDINGS_CACHE[cache_key]
        else:
            embeddings = fetch_embeddings_in_batches(view, embedding_tensor, batch_size=10000)
            # Cache the embeddings for future use
            EMBEDDINGS_CACHE[cache_key] = embeddings
       
        view, scores = vectorstore.python_search_algorithm(
            deeplake_dataset=view,
            query_embedding=query_emb,
            embeddings=embeddings,
            distance_metric=distance_metric.lower(),
            k=k,
        )

        return_data["score"] = scores

    if return_view:
        return view
    else:
        for tensor in return_tensors:
            if tensor == "embedding":
                return_data[tensor] = None  # Skip embedding to save time
                continue

            return_data[tensor] = utils.parse_tensor_return(view[tensor])
        return return_data
