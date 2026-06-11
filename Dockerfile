FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY riemann_zeta_plotly.py riemann_zeta_plotly_2d.py ./

CMD ["sh", "-c", "python riemann_zeta_plotly.py && python riemann_zeta_plotly_2d.py"]
