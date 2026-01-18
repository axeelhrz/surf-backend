#!/bin/bash
# Script de inicio para Railway
# Usa la variable PORT de Railway o 8000 por defecto

PORT=${PORT:-8000}
exec uvicorn main:app --host 0.0.0.0 --port $PORT