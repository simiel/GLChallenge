# Use Python 3.11 Windows Server Core image
FROM python:3.11-windowsservercore

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Add debugging information
RUN powershell -Command "Write-Host 'Contents of /app:'; Get-ChildItem C:\app"
RUN powershell -Command "Write-Host 'Contents of /app/models:'; Get-ChildItem C:\app\models"

# Run the application with Waitress
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "app:app"] 