from .client import Client, create_response_model
from .local import Local
from .openrouter import OpenRouter
from .outlines import Outlines

<<<<<<< HEAD

def get_client(provider: str, model: str, **kwargs):
    if provider == "local":
        return Local(model=model, **kwargs)
    if provider == "openrouter":
        return OpenRouter(model=model, **kwargs)
    if provider == "outlines":
        return Outlines(model=model, **kwargs)

    return None


async def execute_model(
    model,
    queries,
    output_dir: str,
    record_time=False,
    batch_size=100
):
    """
    Executes a model on a list of queries in batches and saves the results to the output directory.
    """

    os.makedirs(output_dir, exist_ok=True)

    async def process_and_save(query):
        logger.info(f"Executing {model.name} on feature layer {query.record.feature}")
        start_time = time.time()
        result = await model(query)
        end_time = time.time()
        filename = f"{query.record.feature}.txt"
        filepath = os.path.join(output_dir, filename)
        if record_time:
            result = {
                "result": result,
                "time": end_time - start_time
            }
            
        async with aiofiles.open(filepath, mode='wb') as f:
            await f.write(orjson.dumps(result))
        logger.info(f"Saved result to {filepath}")

    async def process_batch(batch):
        tasks = [process_and_save(query) for query in batch]
        await asyncio.gather(*tasks)

    # Process queries in batches
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        await process_batch(batch)
        logger.info(f"Completed batch {i//batch_size + 1} of {(len(queries)-1)//batch_size + 1}")
        
=======
__all__ = ["Client", "create_response_model", "Local", "OpenRouter", "Outlines"]
>>>>>>> d3b81f50d676295bd88d0e499194b44c40c02b13
