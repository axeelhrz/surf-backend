"""
Endpoints de Stripe para manejo de pagos
"""
import io
import os
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from payment_models import CheckoutRequest
from payment_service import PaymentService

# STORAGE_DIR (mismo que main.py, sin importar main para evitar imports circulares)
BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "photos_storage")))


def _resolve_folder_path(storage_dir: Path, folder_name: str) -> Optional[Path]:
    """Resuelve el nombre de carpeta de forma insensible a mayúsculas/minúsculas."""
    direct = storage_dir / folder_name
    if direct.exists() and direct.is_dir():
        return direct
    name_lower = folder_name.lower()
    for p in storage_dir.iterdir():
        if p.is_dir() and p.name.lower() == name_lower:
            return p
    return None


router = APIRouter(prefix="/stripe", tags=["stripe"])

@router.get("/config")
async def get_stripe_config():
    """Obtiene la configuración pública de Stripe"""
    try:
        publishable_key = PaymentService.get_stripe_publishable_key()
        
        if not publishable_key:
            raise HTTPException(
                status_code=500,
                detail="Stripe no está configurado. Verifica las variables de entorno."
            )
        
        return {
            "status": "success",
            "publishable_key": publishable_key
        }
    except Exception as e:
        print(f"❌ Error obteniendo configuración de Stripe: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.post("/checkout")
async def create_checkout_session(request: CheckoutRequest):
    """
    Crea una sesión de checkout en Stripe
    
    Args:
        request: CheckoutRequest con items, email y nombre del cliente
        
    Returns:
        Dict con session_id y url de checkout
    """
    try:
        if not request.items or len(request.items) == 0:
            raise HTTPException(
                status_code=400,
                detail="El carrito está vacío"
            )
        
        # Crear sesión de checkout
        session_data = PaymentService.create_checkout_session(
            items=request.items,
            customer_email=request.customer_email,
            customer_name=request.customer_name
        )
        
        return {
            "status": "success",
            "session_id": session_data["session_id"],
            "url": session_data["url"],
            "client_secret": session_data.get("client_secret")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error creando sesión de checkout: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    Obtiene detalles de una sesión de checkout
    
    Args:
        session_id: ID de la sesión de Stripe
        
    Returns:
        Dict con detalles de la sesión
    """
    try:
        session_details = PaymentService.get_session_details(session_id)
        
        return {
            "status": "success",
            "session": session_details
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error obteniendo detalles de sesión: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.post("/confirm-payment")
async def confirm_payment(request: CheckoutRequest, session_id: str):
    """
    Confirma un pago completado
    
    Args:
        request: CheckoutRequest con items y datos del cliente
        session_id: ID de la sesión de Stripe
        
    Returns:
        Dict con confirmación de pago
    """
    try:
        if not session_id:
            raise HTTPException(
                status_code=400,
                detail="session_id es requerido"
            )
        
        # Confirmar pago
        confirmation = PaymentService.confirm_payment(
            session_id=session_id,
            items=request.items,
            customer_email=request.customer_email,
            customer_name=request.customer_name
        )
        
        return confirmation
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error confirmando pago: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/payments")
async def get_all_payments():
    """Obtiene todos los registros de pago"""
    try:
        payments = PaymentService.get_all_payments()
        
        return {
            "status": "success",
            "payments": payments,
            "total": len(payments)
        }
        
    except Exception as e:
        print(f"❌ Error obteniendo pagos: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/payment/{payment_id}")
async def get_payment(payment_id: str):
    """Obtiene un registro de pago específico"""
    try:
        payment = PaymentService.get_payment_by_id(payment_id)
        
        if not payment:
            raise HTTPException(
                status_code=404,
                detail=f"Pago '{payment_id}' no encontrado"
            )
        
        return {
            "status": "success",
            "payment": payment
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error obteniendo pago: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/success-details")
async def get_success_details(session_id: str = None, test: str = None):
    """
    Tras el pago, devuelve el pago y los items para la página de descargas.
    - session_id: ID de la sesión de Stripe (redirect de éxito).
    - test=1: devuelve datos de prueba para ver la sección de descargas sin pago real.
    """
    try:
        if test == "1":
            data = PaymentService.get_test_success_details()
            return {"status": "success", **data}
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id o test=1 requerido")
        data = PaymentService.get_success_details(session_id)
        if not data:
            raise HTTPException(
                status_code=404,
                detail="Pago no encontrado o no completado"
            )
        return {"status": "success", **data}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error success-details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download-zip")
async def download_zip(session_id: str = Query(..., description="ID de la sesión de Stripe")):
    """
    Descarga un ZIP con todas las fotos compradas (sin marca de agua).
    Requiere que el pago esté completado.
    """
    try:
        data = PaymentService.get_success_details(session_id)
        if not data:
            raise HTTPException(
                status_code=404,
                detail="Pago no encontrado o no completado"
            )
        items = data.get("items", [])
        if not items:
            raise HTTPException(
                status_code=400,
                detail="No hay fotos para descargar"
            )

        buffer = io.BytesIO()
        seen = set()  # Evitar duplicados si hay filenames repetidos en distintas carpetas

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in items:
                school = item.get("school", "")
                date = item.get("date", "")
                filename = item.get("filename", "")
                if not school or not filename:
                    continue

                folder_path = _resolve_folder_path(STORAGE_DIR, school)
                if folder_path is None:
                    continue
                if date:
                    photo_path = folder_path / date / filename
                else:
                    photo_path = folder_path / filename

                if not photo_path.exists() or not photo_path.is_file():
                    continue

                # Nombre único en el ZIP: school_date_filename o school_filename
                arcname = f"{school}_{date}_{filename}".replace("__", "_").strip("_") if date else f"{school}_{filename}"
                if arcname in seen:
                    idx = 1
                    base, ext = arcname.rsplit(".", 1) if "." in arcname else (arcname, "")
                    while arcname in seen:
                        arcname = f"{base}_{idx}.{ext}" if ext else f"{base}_{idx}"
                        idx += 1
                seen.add(arcname)

                zf.write(photo_path, arcname)

        if len(seen) == 0:
            raise HTTPException(
                status_code=404,
                detail="No se encontraron archivos de foto en el almacenamiento"
            )

        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="surf-photos.zip"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error download-zip: {e}")
        raise HTTPException(status_code=500, detail=str(e))