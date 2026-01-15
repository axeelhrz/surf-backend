import os
import stripe
from dotenv import load_dotenv
from pathlib import Path

# Cargar variables de entorno
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Configurar Stripe
STRIPE_SECRET_KEY = os.getenv('STRIPE_SECRET_KEY', '')
STRIPE_PUBLISHABLE_KEY = os.getenv('STRIPE_PUBLISHABLE_KEY', '')
STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET', '')

# URLs
FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:3000')
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')

# Inicializar Stripe
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY
    print("✅ Stripe configurado correctamente")
else:
    print("⚠️ STRIPE_SECRET_KEY no configurada. Los pagos no funcionarán.")

# Precios de productos
PHOTO_PRICE_CENTS = 2999  # $29.99 en centavos
PHOTO_PRICE_DOLLARS = 29.99

# Configuración de impuestos
TAX_RATE = 0.10  # 10%