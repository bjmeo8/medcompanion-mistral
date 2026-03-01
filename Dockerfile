# Utilise une image de base Python (par exemple, une version slim pour plus de légèreté)
FROM python:3.10-slim-bookworm

# Met à jour les paquets et installe les outils de base et libmagic
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

# Définit le répertoire de travail dans le conteneur
WORKDIR /app

# Copie les fichiers de dépendances (requirements.txt)
COPY requirements.txt ./

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie le reste de ton code source
COPY . .

# Crée les répertoires d'exécution attendus (DATA, uploads, etc.)
RUN mkdir -p /app/DATA

# Déclare le dossier DATA comme volume pour persister les captures entre redéploiements
VOLUME ["/app/DATA"]

# Expose le port sur lequel ton app va tourner (par défaut 8000 pour uvicorn)
EXPOSE 8000

# Commande par défaut pour lancer l'API FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
