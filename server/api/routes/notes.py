import asyncio

from fastapi import APIRouter, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from fastapi.responses import JSONResponse

from ...core.blocking import blocking_work_pool
from ...core.config import settings
from ...core.rate_limit import consume
from ...deps import get_current_user
from ...models import User
from ...schemas import MeetingNoteRequest
from ...services.meeting_source import MeetingError, load_meeting
from ...services.openwebui_notes import save_recap

class NotesRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def without_secret_validation_input(request):
            try:
                return await handler(request)
            except RequestValidationError:
                # Pydantic model errors can otherwise echo the entire input,
                # including the per-operation bearer token.
                return JSONResponse({'error': 'invalid_notes_request'}, status_code=422)

        return without_secret_validation_input


router = APIRouter(route_class=NotesRoute)


def _save(payload: MeetingNoteRequest, user: User) -> dict:
    snapshot = load_meeting(user_id=user.id, runtime_session_id=payload.runtimeSessionId,
                            history_id=payload.historyId)
    return save_recap(snapshot, user_id=user.id, email=user.email,
                      token=payload.token.get_secret_value(), title=payload.title)


@router.post('/api/meeting/notes')
async def save_meeting_note(payload: MeetingNoteRequest, user: User = Depends(get_current_user)):
    allowed = await asyncio.to_thread(consume, bucket='notes', subject=f'user:{user.id}',
                                      limit=settings.costly_api_rate_limit_requests,
                                      window_seconds=settings.costly_api_rate_limit_window_seconds)
    if not allowed:
        return JSONResponse({'error': 'rate_limit_exceeded'}, status_code=429)
    try:
        result = await blocking_work_pool.run('llm', _save, payload, user)
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})
    except MeetingError as exc:
        return JSONResponse({'error': exc.code}, status_code=exc.status_code,
                            headers={'Cache-Control': 'no-store'})
