import asyncio, aiohttp


async def create_tasks(loop, paras, func):
    tasks = []
    async with aiohttp.ClientSession(loop=loop) as session:
        for para in paras:
            tasks.append(asyncio.ensure_future(func(session, **para)))
        await asyncio.gather(*tasks)
    return tasks


def create_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop = asyncio.get_event_loop()
    return loop


