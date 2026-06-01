import asyncio
import websockets
import json

connected = None

async def handler(websocket):
    global connected
    connected = websocket
    print("游戏已连接！")
    try:
        async for message in websocket:
            data = json.loads(message)
            print(f"收到状态: {data}")
            # 暂时回传一个空动作（什么都不做）
            action = {"action": 0}
            await websocket.send(json.dumps(action))
    except websockets.exceptions.ConnectionClosed:
        print("游戏断开连接")
        connected = None

async def main():
    print("启动 WebSocket Server，等待游戏连接...")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # 永远运行

if __name__ == "__main__":
    asyncio.run(main())
