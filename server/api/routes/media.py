from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...core.blocking import blocking_work_pool
from ...core.config import settings
from ...core.rate_limit import consume
from ...deps import get_current_user
from ...models import User
from ...services.media_import import MAX_UPLOAD_BYTES, import_events, media_format
from ...services.meeting_source import MeetingError
from ...services.meeting_stream_response import stream_events

router = APIRouter()


def _allow(bucket: str, user_id: int):
    return consume(bucket=bucket, subject=f'user:{user_id}',
                   limit=settings.costly_api_rate_limit_requests,
                   window_seconds=settings.costly_api_rate_limit_window_seconds)


@router.post('/api/media/transcribe')
async def transcribe_media(request: Request, filename: str = Query(max_length=240),
                           language: str = Query(default='ja', pattern='^(ja|en|auto|)$'),
                           prompt: str = Query(default='', max_length=750),
                           user: User = Depends(get_current_user)):
    """Stream a bounded binary upload, then SSE progress and a completed runtime."""
    resources = request.app.state.runtime_resources
    if resources.transcriber_factory is None:
        return JSONResponse({'error': 'asr_not_ready'}, status_code=503)
    try:
        media_format(filename)
    except MeetingError as exc:
        return JSONResponse({'error': exc.code}, status_code=exc.status_code)
    length = request.headers.get('content-length')
    if length:
        try:
            if int(length) > MAX_UPLOAD_BYTES:
                return JSONResponse({'error': 'media_too_large'}, status_code=413)
        except ValueError:
            return JSONResponse({'error': 'invalid_content_length'}, status_code=400)
    user_id = user.id
    if not await asyncio.to_thread(_allow, 'media_import', user_id):
        return JSONResponse({'error': 'rate_limit_exceeded'}, status_code=429)
    temporary = tempfile.NamedTemporaryFile(prefix='whistx-upload-', suffix='.media', delete=False)
    path = Path(temporary.name)
    try:
        count = 0
        async with asyncio.timeout(180):
            async for chunk in request.stream():
                count += len(chunk)
                if count > MAX_UPLOAD_BYTES:
                    raise MeetingError('media_too_large', 413)
                await blocking_work_pool.run('artifact', temporary.write, chunk)
        if not count:
            raise MeetingError('media_empty_file', 400)
    except (MeetingError, TimeoutError) as exc:
        path.unlink(missing_ok=True)
        return JSONResponse({'error': exc.code if isinstance(exc, MeetingError) else 'media_upload_timeout'},
                            status_code=exc.status_code if isinstance(exc, MeetingError) else 408)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        temporary.close()

    async def events():
        try:
            async for event in stream_events(lambda cancelled: import_events(path,
                    filename=filename, language=language, prompt=prompt, user_id=user_id,
                    cancelled=cancelled, transcriber_factory=resources.transcriber_factory,
                    observer=resources.observer, allow_request=lambda: _allow('asr', user_id)), pool='asr'):
                yield event
        finally:
            # Also clean uploads when the worker queue times out before starting.
            path.unlink(missing_ok=True)
    return StreamingResponse(events(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})
