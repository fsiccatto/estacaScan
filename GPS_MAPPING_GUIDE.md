# 🗺️ Mapeo GPS de Sectores - Guía de Implementación

## ✅ Lo que agregué al Schema

### Campos GPS en tabla `sectors`:

```sql
-- Coordenadas del centro del sector
center_lat DECIMAL(10, 8)  -- Ej: -38.9516
center_lng DECIMAL(11, 8)  -- Ej: -68.0591

-- Polígono para delimitar el área (opcional)
boundary_polygon JSONB  -- [{lat, lng}, {lat, lng}, ...]

-- Área calculada
area_hectares DECIMAL(10, 4)  -- Ej: 2.5 hectáreas
```

---

## 📍 Cómo Funciona

### Opción 1: Punto Simple (Más fácil)
Solo marcas el **centro** del sector:

```json
{
  "name": "Sector Norte",
  "center_lat": -38.9516,
  "center_lng": -68.0591
}
```

**Uso:**
- Ver sectores en mapa como pins
- Navegar al sector (abrir Google Maps con coordenadas)
- Distancia entre sectores

---

### Opción 2: Polígono (Más preciso)
Marcas los **límites** del sector:

```json
{
  "name": "Sector Norte",
  "boundary_polygon": [
    {"lat": -38.9510, "lng": -68.0580},
    {"lat": -38.9520, "lng": -68.0580},
    {"lat": -38.9520, "lng": -68.0600},
    {"lat": -38.9510, "lng": -68.0600}
  ],
  "area_hectares": 2.5
}
```

**Uso:**
- Ver áreas exactas en mapa
- Calcular superficie automáticamente
- Visualizar límites

---

## 🛠️ Implementación Frontend (Fase Futura)

### Librería recomendada: **Leaflet** (gratis, open source)

```html
<!-- En HTML -->
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<div id="map" style="height: 600px;"></div>
```

```javascript
// Inicializar mapa
const map = L.map('map').setView([-38.9516, -68.0591], 13);

// Agregar tiles (OpenStreetMap gratis)
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

// Marcar sectores
sectors.forEach(sector => {
  if (sector.center_lat && sector.center_lng) {
    L.marker([sector.center_lat, sector.center_lng])
      .bindPopup(`<b>${sector.name}</b><br>${sector.code}`)
      .addTo(map);
  }
  
  // Si tiene polígono
  if (sector.boundary_polygon) {
    const coords = sector.boundary_polygon.map(p => [p.lat, p.lng]);
    L.polygon(coords, {color: 'green'})
      .bindPopup(sector.name)
      .addTo(map);
  }
});
```

---

## 📱 Cómo Capturar Coordenadas

### Método 1: Desde el móvil (recomendado)
1. Ve al sector con tu celular
2. Abre Google Maps
3. Mantén presionado el punto → copia coordenadas
4. Ingresas en la app

### Método 2: Desde Google Maps Web
1. Busca tu finca en maps.google.com
2. Click derecho en el sector → "¿Qué hay aquí?"
3. Aparecen las coordenadas
4. Click para copiar

### Método 3: Geolocalización del navegador (automático)
```javascript
navigator.geolocation.getCurrentPosition(pos => {
  const lat = pos.coords.latitude;
  const lng = pos.coords.longitude;
  // Guardar automáticamente
});
```

---

## 🎯 Features que Habilita

### 1. Mapa de la Finca
Ver todos los sectores en un mapa interactivo.

### 2. Navegación
Botón "Ir al sector" → Abre Google Maps con direcciones.

### 3. Producción por Zona
Heatmap: ¿Qué sectores producen más?

### 4. Cálculo de Áreas
Si usas polígonos, calcula hectáreas automáticamente.

### 5. Análisis Geoespacial
- ¿Qué sector está más lejos?
- ¿Cuál es más grande?
- Rutas óptimas de cosecha

---

## 📊 Ejemplo de Datos Completos

```sql
INSERT INTO sectors (name, code, location, center_lat, center_lng, area_hectares) 
VALUES (
  'Sector Norte - Malbec',
  'SEC-N-MAL',
  'Parcela 1, lindante con ruta',
  -38.9516,
  -68.0591,
  3.2
);
```

Con polígono:
```sql
INSERT INTO sectors (name, code, boundary_polygon) 
VALUES (
  'Sector Sur - Cabernet',
  'SEC-S-CAB',
  '[
    {"lat": -38.9520, "lng": -68.0600},
    {"lat": -38.9530, "lng": -68.0600},
    {"lat": -38.9530, "lng": -68.0620},
    {"lat": -38.9520, "lng": -68.0620}
  ]'::jsonb
);
```

---

## 🚀 Implementación por Fases

### Fase 2A (Actual)
✅ Schema listo con campos GPS
✅ Puedes ingresar coordenadas manualmente

### Fase 2B (Próxima)
- UI para ingresar coordenadas fácilmente
- Botón "Usar mi ubicación actual"

### Fase 3
- Mapa interactivo (Leaflet)
- Visualización de todos los sectores
- Click en sector → ver historial

### Fase 4 (Avanzada)
- Dibujar polígonos desde la app
- Cálculo automático de área
- Heatmaps de producción

---

## 💡 Alternativas si no quieres mapas complejos

Si solo quieres **algo simple**:
- Solo guarda `center_lat`, `center_lng`
- Botón "Ir" → Abre `google.com/maps?q=lat,lng`
- No necesitas visualización en la app

**Beneficio:** Navegación GPS sin complejidad técnica.

---

## ⚠️ Consideraciones

- **Precisión:** GPS móvil ±5-10 metros (suficiente para sectores)
- **Privacidad:** Coordenadas solo se ven en tu BD, no son públicas
- **Offline:** Guardar coords funciona offline, mapas requieren internet

---

¿Quieres que avancemos con lo básico primero (schema actual) y dejamos el mapa visual para después, o preferís agregar el mapa de una vez?
