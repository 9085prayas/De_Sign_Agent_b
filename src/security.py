import os
import logging
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
security_scheme = HTTPBearer()

DESCOPE_PROJECT_ID = os.environ.get("DESCOPE_PROJECT_ID")
descope_client = None

try:
    if DESCOPE_PROJECT_ID:
        from descope import DescopeClient  # optional dependency
        descope_client = DescopeClient(project_id=DESCOPE_PROJECT_ID)
        logging.info("Descope client initialized.")
    else:
        logging.warning("DESCOPE_PROJECT_ID not set. Security checks are disabled for local development.")
except Exception as e:
    logging.warning("Descope not available (%s). Running unsecured for local dev.", e)
    descope_client = None

def require_scope(required_scope: str):
    def guard(token: str = Depends(security_scheme)):
        if not descope_client:
            # Local/dev: allow through
            return {"dev": True, "scope_bypass": required_scope}
        try:
            jwt_response = descope_client.validate_permissions(
                session_token=token.credentials,
                permissions=[required_scope],
            )
            return jwt_response
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Invalid or expired token: {e}")
    return guard