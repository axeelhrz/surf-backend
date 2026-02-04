"""
Endpoints de Stripe para manejo de pagos
"""
from fastapi import APIRouter, HTTPException
from payment_models import CheckoutRequest
from payment_service import PaymentService

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