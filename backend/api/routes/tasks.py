"""
SSE task progress stream.
GET /api/tasks/{task_id}/stream → Server-Sent Events
"""

import asyncio
import json
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from api.routes.hype_cycle import get_task

router = APIRouter()


@router.get("/{task_id}/stream")
async def stream_task(task_id: str):
    """Stream task progress as Server-Sent Events."""

    async def event_generator():
        while True:
            task = get_task(task_id)

            if not task:
                yield {
                    "data": json.dumps({
                        "task_id": task_id,
                        "status": "failed",
                        "progress": 0,
                        "message": "Task not found",
                        "error": "Task not found",
                    })
                }
                break

            payload = {
                "task_id": task_id,
                "status": task.get("status", "queued"),
                "progress": task.get("progress", 0),
                "message": task.get("message", ""),
            }

            if task.get("result"):
                payload["result"] = task["result"]

            if task.get("error"):
                payload["error"] = task["error"]

            yield {"data": json.dumps(payload)}

            if task.get("status") in ("completed", "failed"):
                break

            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())
