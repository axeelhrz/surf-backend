"""
Endpoints específicos para el panel de administración
"""
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
from payment_service import PaymentService

router = APIRouter(prefix="/admin", tags=["admin"])

# Directorio base
BASE_DIR = Path(__file__).parent.absolute()
STORAGE_DIR = BASE_DIR / "photos_storage"
PAYMENTS_DIR = BASE_DIR / "payments_storage"

@router.get("/dashboard/stats")
async def get_dashboard_stats():
    """
    Obtiene estadísticas generales para el dashboard
    """
    try:
        # Contar carpetas
        folders = [f for f in STORAGE_DIR.iterdir() if f.is_dir()]
        total_folders = len(folders)
        
        # Contar fotos
        total_photos = 0
        for folder in folders:
            photos = [f for f in folder.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}]
            total_photos += len(photos)
        
        # Obtener pagos
        payments = PaymentService.get_all_payments()
        total_revenue = sum(payment.get('amount', 0) for payment in payments)
        total_transactions = len(payments)
        
        # Calcular cambios (últimos 7 días vs anteriores)
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        two_weeks_ago = now - timedelta(days=14)
        
        recent_payments = [p for p in payments if datetime.fromisoformat(p.get('created_at', '2000-01-01')) > week_ago]
        previous_payments = [p for p in payments if two_weeks_ago < datetime.fromisoformat(p.get('created_at', '2000-01-01')) <= week_ago]
        
        recent_revenue = sum(p.get('amount', 0) for p in recent_payments)
        previous_revenue = sum(p.get('amount', 0) for p in previous_payments)
        
        revenue_change = 0
        if previous_revenue > 0:
            revenue_change = ((recent_revenue - previous_revenue) / previous_revenue) * 100
        
        return {
            "status": "success",
            "stats": {
                "total_folders": total_folders,
                "total_photos": total_photos,
                "total_revenue": round(total_revenue, 2),
                "total_transactions": total_transactions,
                "recent_transactions": len(recent_payments),
                "revenue_change_percentage": round(revenue_change, 2)
            }
        }
    except Exception as e:
        print(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/dashboard/activity")
async def get_recent_activity(limit: int = Query(10, ge=1, le=50)):
    """
    Obtiene actividad reciente del sistema
    """
    try:
        activities = []
        
        # Obtener pagos recientes
        payments = PaymentService.get_all_payments()
        for payment in sorted(payments, key=lambda x: x.get('created_at', ''), reverse=True)[:limit]:
            activities.append({
                "type": "Pago",
                "description": f"Transacción de {payment.get('customer_name', 'Cliente')} - ${payment.get('amount', 0):.2f}",
                "date": payment.get('created_at'),
                "status": "Completado" if payment.get('status') == 'completed' else "Pendiente"
            })
        
        # Obtener carpetas recientes
        folders = sorted(
            [f for f in STORAGE_DIR.iterdir() if f.is_dir()],
            key=lambda x: x.stat().st_ctime,
            reverse=True
        )[:5]
        
        for folder in folders:
            metadata_path = folder / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                activities.append({
                    "type": "Carpeta",
                    "description": f"Carpeta '{folder.name}' creada",
                    "date": metadata.get('created_at'),
                    "status": "Completado"
                })
        
        # Ordenar por fecha
        activities.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        return {
            "status": "success",
            "activities": activities[:limit]
        }
    except Exception as e:
        print(f"Error obteniendo actividad: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/payments/summary")
async def get_payments_summary(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Obtiene resumen de pagos con filtros opcionales
    """
    try:
        payments = PaymentService.get_all_payments()
        
        # Filtrar por fechas si se proporcionan
        if start_date:
            start = datetime.fromisoformat(start_date)
            payments = [p for p in payments if datetime.fromisoformat(p.get('created_at', '2000-01-01')) >= start]
        
        if end_date:
            end = datetime.fromisoformat(end_date)
            payments = [p for p in payments if datetime.fromisoformat(p.get('created_at', '2000-01-01')) <= end]
        
        # Calcular resumen
        total_amount = sum(p.get('amount', 0) for p in payments)
        total_transactions = len(payments)
        avg_transaction = total_amount / total_transactions if total_transactions > 0 else 0
        
        # Agrupar por estado
        by_status = {}
        for payment in payments:
            status = payment.get('status', 'unknown')
            if status not in by_status:
                by_status[status] = {"count": 0, "amount": 0}
            by_status[status]["count"] += 1
            by_status[status]["amount"] += payment.get('amount', 0)
        
        # Agrupar por mes
        by_month = {}
        for payment in payments:
            date = datetime.fromisoformat(payment.get('created_at', '2000-01-01'))
            month_key = date.strftime('%Y-%m')
            if month_key not in by_month:
                by_month[month_key] = {"count": 0, "amount": 0}
            by_month[month_key]["count"] += 1
            by_month[month_key]["amount"] += payment.get('amount', 0)
        
        return {
            "status": "success",
            "summary": {
                "total_amount": round(total_amount, 2),
                "total_transactions": total_transactions,
                "average_transaction": round(avg_transaction, 2),
                "by_status": by_status,
                "by_month": dict(sorted(by_month.items(), reverse=True))
            }
        }
    except Exception as e:
        print(f"Error obteniendo resumen de pagos: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/reports/photos-by-folder")
async def get_photos_by_folder():
    """
    Obtiene reporte de fotos por carpeta
    """
    try:
        folders = [f for f in STORAGE_DIR.iterdir() if f.is_dir()]
        
        report = []
        for folder in folders:
            photos = [f for f in folder.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}]
            total_size = sum(f.stat().st_size for f in photos)
            
            report.append({
                "folder_name": folder.name,
                "photo_count": len(photos),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(folder.stat().st_ctime).isoformat()
            })
        
        # Ordenar por cantidad de fotos
        report.sort(key=lambda x: x['photo_count'], reverse=True)
        
        return {
            "status": "success",
            "report": report,
            "total_folders": len(report),
            "total_photos": sum(r['photo_count'] for r in report),
            "total_size_mb": round(sum(r['total_size_mb'] for r in report), 2)
        }
    except Exception as e:
        print(f"Error generando reporte: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/reports/revenue-by-period")
async def get_revenue_by_period(
    period: str = Query("month", regex="^(day|week|month|year)$")
):
    """
    Obtiene reporte de ingresos por período
    """
    try:
        payments = PaymentService.get_all_payments()
        
        # Agrupar por período
        revenue_by_period = {}
        
        for payment in payments:
            date = datetime.fromisoformat(payment.get('created_at', '2000-01-01'))
            
            if period == "day":
                key = date.strftime('%Y-%m-%d')
            elif period == "week":
                key = f"{date.year}-W{date.isocalendar()[1]:02d}"
            elif period == "month":
                key = date.strftime('%Y-%m')
            else:  # year
                key = str(date.year)
            
            if key not in revenue_by_period:
                revenue_by_period[key] = {
                    "period": key,
                    "amount": 0,
                    "transactions": 0,
                    "items_sold": 0
                }
            
            revenue_by_period[key]["amount"] += payment.get('amount', 0)
            revenue_by_period[key]["transactions"] += 1
            revenue_by_period[key]["items_sold"] += payment.get('items_count', 0)
        
        # Convertir a lista y ordenar
        report = sorted(revenue_by_period.values(), key=lambda x: x['period'], reverse=True)
        
        # Redondear montos
        for item in report:
            item["amount"] = round(item["amount"], 2)
        
        return {
            "status": "success",
            "period": period,
            "report": report,
            "total_periods": len(report)
        }
    except Exception as e:
        print(f"Error generando reporte de ingresos: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/settings")
async def get_settings():
    """
    Obtiene configuración del sistema
    """
    try:
        from stripe_config import (
            STRIPE_PUBLISHABLE_KEY,
            PHOTO_PRICE_DOLLARS,
            TAX_RATE,
            FRONTEND_URL,
            BACKEND_URL
        )
        
        return {
            "status": "success",
            "settings": {
                "stripe_publishable_key": STRIPE_PUBLISHABLE_KEY,
                "photo_price": PHOTO_PRICE_DOLLARS,
                "tax_rate": TAX_RATE * 100,  # Convertir a porcentaje
                "frontend_url": FRONTEND_URL,
                "backend_url": BACKEND_URL
            }
        }
    except Exception as e:
        print(f"Error obteniendo configuración: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")