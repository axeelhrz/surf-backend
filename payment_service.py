import os
import stripe
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from payment_models import PhotoItem, PaymentRecord
from stripe_config import (
    STRIPE_SECRET_KEY,
    STRIPE_PUBLISHABLE_KEY,
    FRONTEND_URL,
    BACKEND_URL,
    PHOTO_PRICE_CENTS,
    TAX_RATE,
)

# Directorio para almacenar registros de pagos (env para persistencia en Railway)
PAYMENTS_DIR = Path(os.getenv("PAYMENTS_DIR", str(Path(__file__).parent / "payments_storage")))
try:
    PAYMENTS_DIR.mkdir(exist_ok=True, parents=True)
except OSError as e:
    print(f"⚠️ No se pudo crear PAYMENTS_DIR {PAYMENTS_DIR}: {e}")

class PaymentService:
    """Servicio para manejar pagos con Stripe"""
    
    @staticmethod
    def create_checkout_session(
        items: List[PhotoItem],
        customer_email: str,
        customer_name: str
    ) -> Dict:
        """
        Crea una sesión de checkout en Stripe
        
        Args:
            items: Lista de fotos a comprar
            customer_email: Email del cliente
            customer_name: Nombre del cliente
            
        Returns:
            Dict con session_id y url de checkout
        """
        try:
            if not STRIPE_SECRET_KEY:
                raise ValueError("STRIPE_SECRET_KEY no configurada")
            
            # Preparar line items para Stripe
            line_items = []
            for item in items:
                line_items.append({
                    "price_data": {
                        "currency": "eur",
                        "product_data": {
                            "name": f"Foto: {item.filename}",
                            "description": f"Escuela: {item.school} | Fecha: {item.date}",
                            "images": [item.thumbnail] if item.thumbnail else [],
                        },
                        "unit_amount": int(item.price * 100),  # Convertir a centavos
                    },
                    "quantity": 1,
                })
            
            # Calcular totales
            subtotal = sum(item.price for item in items)
            tax_amount = subtotal * TAX_RATE
            total = subtotal + tax_amount
            
            # Crear sesión de checkout
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=line_items,
                mode="payment",
                customer_email=customer_email,
                success_url=f"{FRONTEND_URL}/checkout/success?session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=f"{FRONTEND_URL}/checkout/cancel",
                metadata={
                    "customer_name": customer_name,
                    "items_count": len(items),
                    "subtotal": str(subtotal),
                    "tax": str(tax_amount),
                    "total": str(total),
                }
            )
            
            # Guardar sesión pendiente para la página de éxito (descargas)
            PaymentService._save_pending_session(session.id, {
                "items": [item.dict() for item in items],
                "customer_email": customer_email,
                "customer_name": customer_name,
            })
            
            return {
                "session_id": session.id,
                "url": session.url,
                "client_secret": session.client_secret,
                "status": "created"
            }
            
        except stripe.error.StripeError as e:
            print(f"❌ Error de Stripe: {e}")
            raise Exception(f"Error creando sesión de pago: {str(e)}")
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            raise Exception(f"Error inesperado: {str(e)}")
    
    @staticmethod
    def get_session_details(session_id: str) -> Dict:
        """
        Obtiene detalles de una sesión de checkout
        
        Args:
            session_id: ID de la sesión de Stripe
            
        Returns:
            Dict con detalles de la sesión
        """
        try:
            if not STRIPE_SECRET_KEY:
                raise ValueError("STRIPE_SECRET_KEY no configurada")
            
            session = stripe.checkout.Session.retrieve(session_id)
            
            return {
                "session_id": session.id,
                "status": session.payment_status,
                "customer_email": session.customer_email,
                "amount_total": session.amount_total / 100 if session.amount_total else 0,
                "currency": session.currency,
                "payment_intent": session.payment_intent,
                "metadata": session.metadata,
            }
            
        except stripe.error.StripeError as e:
            print(f"❌ Error de Stripe: {e}")
            raise Exception(f"Error obteniendo detalles de sesión: {str(e)}")
    
    @staticmethod
    def confirm_payment(
        session_id: str,
        items: List[PhotoItem],
        customer_email: str,
        customer_name: str
    ) -> Dict:
        """
        Confirma un pago y lo registra
        
        Args:
            session_id: ID de la sesión de Stripe
            items: Lista de fotos compradas
            customer_email: Email del cliente
            customer_name: Nombre del cliente
            
        Returns:
            Dict con confirmación de pago
        """
        try:
            # Obtener detalles de la sesión
            session = stripe.checkout.Session.retrieve(session_id)
            
            if session.payment_status != "paid":
                raise Exception(f"Pago no completado. Estado: {session.payment_status}")
            
            # Crear registro de pago
            payment_record = {
                "id": f"PAY_{datetime.now().timestamp()}",
                "stripe_session_id": session.id,
                "stripe_payment_intent_id": session.payment_intent,
                "customer_email": customer_email,
                "customer_name": customer_name,
                "amount": session.amount_total / 100 if session.amount_total else 0,
                "currency": session.currency,
                "status": "completed",
                "items_count": len(items),
                "items": [item.dict() for item in items],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            
            # Guardar registro en archivo JSON
            PaymentService.save_payment_record(payment_record)
            
            return {
                "status": "success",
                "payment_id": payment_record["id"],
                "session_id": session.id,
                "amount": payment_record["amount"],
                "currency": payment_record["currency"],
                "customer_email": customer_email,
                "customer_name": customer_name,
                "items_count": len(items),
                "message": "Pago confirmado exitosamente"
            }
            
        except stripe.error.StripeError as e:
            print(f"❌ Error de Stripe: {e}")
            raise Exception(f"Error confirmando pago: {str(e)}")
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            raise Exception(f"Error inesperado: {str(e)}")
    
    @staticmethod
    def save_payment_record(payment_record: Dict) -> None:
        """
        Guarda un registro de pago en archivo JSON
        
        Args:
            payment_record: Dict con datos del pago
        """
        try:
            # Archivo de registro de pagos
            payments_file = PAYMENTS_DIR / "payments.json"
            
            # Cargar pagos existentes
            payments = []
            if payments_file.exists():
                with open(payments_file, 'r') as f:
                    payments = json.load(f)
            
            # Agregar nuevo pago
            payments.append(payment_record)
            
            # Guardar
            with open(payments_file, 'w') as f:
                json.dump(payments, f, indent=2)
            
            print(f"✅ Pago registrado: {payment_record['id']}")
            
        except Exception as e:
            print(f"❌ Error guardando registro de pago: {e}")
            raise
    
    @staticmethod
    def get_all_payments() -> List[Dict]:
        """
        Obtiene todos los registros de pago
        
        Returns:
            Lista de registros de pago
        """
        try:
            payments_file = PAYMENTS_DIR / "payments.json"
            
            if not payments_file.exists():
                return []
            
            with open(payments_file, 'r') as f:
                payments = json.load(f)
            
            return payments
            
        except Exception as e:
            print(f"❌ Error obteniendo pagos: {e}")
            return []
    
    @staticmethod
    def get_payment_by_id(payment_id: str) -> Optional[Dict]:
        """
        Obtiene un registro de pago por ID
        
        Args:
            payment_id: ID del pago
            
        Returns:
            Dict con datos del pago o None
        """
        try:
            payments = PaymentService.get_all_payments()
            
            for payment in payments:
                if payment["id"] == payment_id:
                    return payment
            
            return None
            
        except Exception as e:
            print(f"❌ Error obteniendo pago: {e}")
            return None
    
    @staticmethod
    def _pending_sessions_path() -> Path:
        return PAYMENTS_DIR / "pending_sessions.json"
    
    @staticmethod
    def _save_pending_session(session_id: str, data: Dict) -> None:
        try:
            path = PaymentService._pending_sessions_path()
            sessions = {}
            if path.exists():
                with open(path, 'r') as f:
                    sessions = json.load(f)
            sessions[session_id] = data
            with open(path, 'w') as f:
                json.dump(sessions, f, indent=2)
        except Exception as e:
            print(f"❌ Error guardando sesión pendiente: {e}")
    
    @staticmethod
    def _load_pending_session(session_id: str) -> Optional[Dict]:
        try:
            path = PaymentService._pending_sessions_path()
            if not path.exists():
                return None
            with open(path, 'r') as f:
                sessions = json.load(f)
            return sessions.get(session_id)
        except Exception as e:
            print(f"❌ Error cargando sesión pendiente: {e}")
            return None
    
    @staticmethod
    def _remove_pending_session(session_id: str) -> None:
        try:
            path = PaymentService._pending_sessions_path()
            if not path.exists():
                return
            with open(path, 'r') as f:
                sessions = json.load(f)
            sessions.pop(session_id, None)
            with open(path, 'w') as f:
                json.dump(sessions, f, indent=2)
        except Exception as e:
            print(f"❌ Error eliminando sesión pendiente: {e}")
    
    @staticmethod
    def get_success_details(session_id: str) -> Optional[Dict]:
        """
        Obtiene los detalles del pago tras el redirect de éxito.
        Verifica que el pago esté completado, crea el registro de pago y devuelve los items para descarga.
        """
        try:
            if not STRIPE_SECRET_KEY:
                return None
            session = stripe.checkout.Session.retrieve(session_id)
            if session.payment_status != "paid":
                return None
            pending = PaymentService._load_pending_session(session_id)
            if not pending:
                # Ya confirmado antes: buscar pago existente por stripe_session_id
                payments = PaymentService.get_all_payments()
                for p in payments:
                    if p.get("stripe_session_id") == session_id:
                        return {"payment": p, "items": p.get("items", [])}
                return None
            items_data = pending.get("items", [])
            items = [PhotoItem(**it) for it in items_data]
            payment_record = {
                "id": f"PAY_{datetime.now().timestamp()}",
                "stripe_session_id": session_id,
                "stripe_payment_intent_id": session.payment_intent,
                "customer_email": pending.get("customer_email", session.customer_email or ""),
                "customer_name": pending.get("customer_name", ""),
                "amount": session.amount_total / 100 if session.amount_total else 0,
                "currency": session.currency or "usd",
                "status": "completed",
                "items_count": len(items),
                "items": items_data,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            PaymentService.save_payment_record(payment_record)
            PaymentService._remove_pending_session(session_id)
            return {"payment": payment_record, "items": items_data}
        except Exception as e:
            print(f"❌ Error get_success_details: {e}")
            return None
    
    @staticmethod
    def get_test_success_details() -> Dict:
        """Devuelve datos de prueba para la página de descargas (sin pago real)."""
        return {
            "payment": {
                "id": "TEST_001",
                "customer_email": "cliente@ejemplo.com",
                "customer_name": "Cliente de prueba",
                "amount": 35,
                "currency": "eur",
                "status": "completed",
                "items_count": 2,
                "created_at": datetime.now().isoformat(),
            },
            "items": [
                {"id": "1", "filename": "foto1.jpg", "school": "LA CABRA SURF SCHOOL", "date": "2025-02-01", "price": 35, "thumbnail": None},
                {"id": "2", "filename": "foto2.jpg", "school": "LA CABRA SURF SCHOOL", "date": "2025-02-01", "price": 15, "thumbnail": None},
            ],
            "backend_url": BACKEND_URL,
        }
    
    @staticmethod
    def get_stripe_publishable_key() -> str:
        """
        Obtiene la clave pública de Stripe
        
        Returns:
            Clave pública de Stripe
        """
        return STRIPE_PUBLISHABLE_KEY