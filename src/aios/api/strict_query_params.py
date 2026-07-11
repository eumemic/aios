"""Fail closed when API callers send undeclared query parameters."""

from fastapi import HTTPException, Request
from fastapi.routing import APIRoute

# Webhook providers commonly append their own query parameters. The opaque token
# authenticates this raw ingest surface, so rejecting those parameters could drop
# otherwise valid external events.
_EXEMPT_PATH_PREFIX = "/v1/triggers/ingest/"


def _declared_query_aliases(route: APIRoute) -> set[str]:
    """Collect public query names, including parameters from dependencies."""
    aliases: set[str] = set()
    pending = [route.dependant]
    while pending:
        dependant = pending.pop()
        aliases.update(param.alias for param in dependant.query_params)
        pending.extend(dependant.dependencies)
    return aliases


async def reject_unknown_query_params(request: Request) -> None:
    """Reject query keys absent from the matched operation's declaration."""
    if request.url.path.startswith(_EXEMPT_PATH_PREFIX):
        return
    route = request.scope.get("route")
    if not isinstance(route, APIRoute):
        return
    unknown = sorted(set(request.query_params) - _declared_query_aliases(route))
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "type": "extra_forbidden",
                    "loc": ["query", name],
                    "msg": "Unknown query parameter",
                    "input": request.query_params.getlist(name),
                }
                for name in unknown
            ],
        )
