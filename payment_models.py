from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class PhotoItem(BaseModel):
    """Modelo para un artículo de foto en el carrito"""
    id: str
    filename: str
    school: str
    date: str
    price: float
    thumbnail: Optional[str] = None

class CheckoutRequest(BaseModel):
    """Modelo para solicitud de checkout"""
    items: List[PhotoItem]
    customer_email: str
    customer_name: str

class CheckoutSession(BaseModel):
    """Modelo para respuesta de sesión de checkout"""
    session_id: str
    client_secret: Optional[str] = None
    url: Optional[str] = None

class PaymentConfirmation(BaseModel):
    """Modelo para confirmación de pago"""
    session_id: str
    payment_intent_id: str
    status: str
    amount: float
    currency: str
    customer_email: str
    customer_name: str
    items: List[PhotoItem]
    created_at: datetime

class PaymentRecord(BaseModel):
    """Modelo para registro de pago en base de datos"""
    id: str
    stripe_session_id: str
    stripe_payment_intent_id: str
    customer_email: str
    customer_name: str
    amount: float
    currency: str = "usd"
    status: str
    items_count: int
    items: List[dict]
    created_at: datetime
    updated_at: datetime