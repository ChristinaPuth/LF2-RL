import asyncio
import websockets
import json

async def handler(websocket):
    print("游戏已连接！")
    count = 0
    async for message in websocket:
        data = json.loads(message)
        if count % 30 == 0:  # 每30帧打印一次
            print(f"t={data['time']} davis_hp={data['davis']['hp']} dennis_hp={data['dennis']['hp']}")
        count += 1
        await websocket.send(json.dumps({"action": 0}))

async def main():
    print("启动中，等待浏览器连接...")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()

asyncio.run(main())
