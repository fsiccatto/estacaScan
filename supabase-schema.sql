-- ===============================================
-- EstacaScan - Supabase Database Schema v2
-- CORREGIDO: Sectores (ubicación) vs Lotes (grupos de estacas)
-- Pagos DIARIOS, no por período
-- ===============================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ===============================================
-- Table: config
-- Configuración global del sistema
-- ===============================================
CREATE TABLE IF NOT EXISTS config (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Precio global por estaca (aplicable a todos los cosechadores)
  price_per_stake DECIMAL(10,2) DEFAULT 0.00,
  
  -- Otras configs futuras
  company_name VARCHAR(255),
  notes TEXT
);

-- Insertar config inicial
INSERT INTO config (price_per_stake, company_name) 
VALUES (0.50, 'Mi Vivero')
ON CONFLICT DO NOTHING;

-- ===============================================
-- Table: harvesters
-- Cosechadores/trabajadores
-- ===============================================
CREATE TABLE IF NOT EXISTS harvesters (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Info personal
  name VARCHAR(255) NOT NULL,
  dni VARCHAR(20) UNIQUE,
  phone VARCHAR(20),
  email VARCHAR(255),
  
  -- Status
  is_active BOOLEAN DEFAULT true,
  
  -- Stats (calculadas desde analysis)
  total_stakes INT DEFAULT 0,
  total_earnings DECIMAL(10,2) DEFAULT 0.00,
  
  -- Metadata
  notes TEXT
);

CREATE INDEX idx_harvesters_active ON harvesters(is_active) WHERE is_active = true;

-- ===============================================
-- Table: sectors
-- Sectores/partes físicas de la finca
-- Con soporte para mapeo GPS
-- ===============================================
CREATE TABLE IF NOT EXISTS sectors (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Info del sector
  name VARCHAR(100) NOT NULL,
  code VARCHAR(50) UNIQUE,
  location TEXT, -- ej: "Norte, cerca del río"
  
  -- Geolocalización (GPS)
  center_lat DECIMAL(10, 8), -- Latitud del centro (-90 a 90)
  center_lng DECIMAL(11, 8), -- Longitud del centro (-180 a 180)
  
  -- Polígono del sector (opcional, para áreas complejas)
  -- Formato: JSON array de puntos [{lat, lng}, {lat, lng}, ...]
  boundary_polygon JSONB,
  
  -- Área calculada (hectáreas)
  area_hectares DECIMAL(10, 4),
  
  -- Status
  is_active BOOLEAN DEFAULT true,
  
  -- Metadata
  notes TEXT
);

CREATE INDEX idx_sectors_active ON sectors(is_active) WHERE is_active = true;
CREATE INDEX idx_sectors_location ON sectors(center_lat, center_lng) WHERE center_lat IS NOT NULL;

-- ===============================================
-- Table: batches
-- Lotes de estacas (agrupadas por variedad y fecha)
-- ===============================================
CREATE TABLE IF NOT EXISTS batches (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Identificación del lote
  code VARCHAR(50) UNIQUE NOT NULL, -- Código único del lote
  variety VARCHAR(100) NOT NULL, -- ej: "Álamo", "Sauce", "Populus nigra"
  harvest_date DATE, -- Fecha de cosecha
  
  -- Estados del lote (workflow)
  status VARCHAR(50) DEFAULT 'cosechado', 
  -- Ejemplos de estados:
  -- - cosechado
  -- - en_procesamiento
  -- - almacenado
  -- - vendido
  -- - descartado
  
  -- Cliente (si está vendido o reservado)
  client_name VARCHAR(255),
  
  -- Stats
  total_stakes INT DEFAULT 0,
  
  -- Metadata
  notes TEXT,
  
  -- Status
  is_active BOOLEAN DEFAULT true
);

CREATE INDEX idx_batches_active ON batches(is_active) WHERE is_active = true;
CREATE INDEX idx_batches_status ON batches(status);
CREATE INDEX idx_batches_variety ON batches(variety);

-- ===============================================
-- Table: analysis
-- Historial de análisis (SIN imágenes)
-- ===============================================
CREATE TABLE IF NOT EXISTS analysis (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Relaciones
  harvester_id UUID REFERENCES harvesters(id) ON DELETE SET NULL,
  sector_id UUID REFERENCES sectors(id) ON DELETE SET NULL,
  batch_id UUID REFERENCES batches(id) ON DELETE SET NULL,
  
  -- Resultados del conteo
  total_confirmed INT NOT NULL,
  ia_base INT NOT NULL,
  manually_accepted INT DEFAULT 0,
  manually_added INT DEFAULT 0,
  rejected INT DEFAULT 0,
  doubts_reviewed INT DEFAULT 0,
  
  -- Metadata de la imagen (NO guardamos la imagen)
  image_width INT,
  image_height INT,
  
  -- Performance
  processing_time_ms INT,
  
  -- Notas del usuario
  notes TEXT
);

-- Indexes para queries comunes
CREATE INDEX idx_analysis_harvester ON analysis(harvester_id);
CREATE INDEX idx_analysis_sector ON analysis(sector_id);
CREATE INDEX idx_analysis_batch ON analysis(batch_id);
CREATE INDEX idx_analysis_created ON analysis(created_at DESC);
CREATE INDEX idx_analysis_date ON analysis(DATE(created_at));

-- ===============================================
-- Table: daily_payments
-- Pagos DIARIOS a cosechadores
-- ===============================================
CREATE TABLE IF NOT EXISTS daily_payments (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Relación
  harvester_id UUID REFERENCES harvesters(id) ON DELETE RESTRICT NOT NULL,
  
  -- Fecha del pago (día específico)
  payment_date DATE NOT NULL,
  
  -- Cálculo
  total_stakes INT NOT NULL,
  price_per_stake DECIMAL(10,2) NOT NULL,
  total_amount DECIMAL(10,2) NOT NULL,
  
  -- Estado
  status VARCHAR(20) DEFAULT 'pending', -- pending, paid, cancelled
  paid_at TIMESTAMPTZ,
  payment_method VARCHAR(50), -- efectivo, transferencia, etc.
  
  -- Notas
  notes TEXT,
  
  -- Constraint: Un solo pago por cosechador por día
  UNIQUE(harvester_id, payment_date)
);

-- Indexes
CREATE INDEX idx_daily_payments_harvester ON daily_payments(harvester_id);
CREATE INDEX idx_daily_payments_date ON daily_payments(payment_date);
CREATE INDEX idx_daily_payments_status ON daily_payments(status);

-- ===============================================
-- Row Level Security (RLS)
-- Por ahora DISABLED para desarrollo
-- ===============================================
ALTER TABLE config DISABLE ROW LEVEL SECURITY;
ALTER TABLE harvesters DISABLE ROW LEVEL SECURITY;
ALTER TABLE sectors DISABLE ROW LEVEL SECURITY;
ALTER TABLE batches DISABLE ROW LEVEL SECURITY;
ALTER TABLE analysis DISABLE ROW LEVEL SECURITY;
ALTER TABLE daily_payments DISABLE ROW LEVEL SECURITY;

-- ===============================================
-- Functions & Triggers
-- ===============================================

-- Función para actualizar updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers para updated_at
CREATE TRIGGER update_config_updated_at BEFORE UPDATE ON config
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_harvesters_updated_at BEFORE UPDATE ON harvesters
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sectors_updated_at BEFORE UPDATE ON sectors
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_batches_updated_at BEFORE UPDATE ON batches
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_daily_payments_updated_at BEFORE UPDATE ON daily_payments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ===============================================
-- Sample Data
-- ===============================================

-- Cosechadores de ejemplo
INSERT INTO harvesters (name, dni) VALUES
  ('Juan Pérez', '12345678'),
  ('María González', '87654321')
ON CONFLICT (dni) DO NOTHING;

-- Sectores de ejemplo
INSERT INTO sectors (name, code, location) VALUES
  ('Sector Norte', 'SEC-N', 'Parcela 1, zona alta'),
  ('Sector Sur', 'SEC-S', 'Parcela 2, cerca del arroyo')
ON CONFLICT (code) DO NOTHING;

-- Lotes de ejemplo (estacas de vid)
INSERT INTO batches (code, variety, harvest_date, status) VALUES
  ('LOT-2024-MAL-001', 'Malbec', '2024-01-15', 'almacenado'),
  ('LOT-2024-CAB-001', 'Cabernet Sauvignon', '2024-01-16', 'cosechado'),
  ('LOT-2024-TOR-001', 'Torrontés', '2024-01-17', 'en_procesamiento')
ON CONFLICT (code) DO NOTHING;

-- ===============================================
-- Helper Views
-- ===============================================

-- Vista: Análisis con nombres legibles
CREATE OR REPLACE VIEW v_analysis_detailed AS
SELECT 
  a.id,
  a.created_at,
  h.name as harvester_name,
  s.name as sector_name,
  s.code as sector_code,
  b.code as batch_code,
  b.variety as batch_variety,
  a.total_confirmed,
  a.ia_base,
  a.manually_accepted,
  a.manually_added,
  a.rejected,
  a.notes
FROM analysis a
LEFT JOIN harvesters h ON a.harvester_id = h.id
LEFT JOIN sectors s ON a.sector_id = s.id
LEFT JOIN batches b ON a.batch_id = b.id
ORDER BY a.created_at DESC;

-- Vista: Pagos pendientes por día
CREATE OR REPLACE VIEW v_pending_payments AS
SELECT 
  dp.id,
  dp.payment_date,
  h.name as harvester_name,
  dp.total_stakes,
  dp.price_per_stake,
  dp.total_amount,
  dp.status
FROM daily_payments dp
JOIN harvesters h ON dp.harvester_id = h.id
WHERE dp.status = 'pending'
ORDER BY dp.payment_date DESC;

-- ===============================================
-- Verificación
-- ===============================================

-- Query para verificar tablas creadas
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name IN ('config', 'harvesters', 'sectors', 'batches', 'analysis', 'daily_payments')
ORDER BY table_name;

-- Verificar vistas
SELECT table_name
FROM information_schema.views
WHERE table_schema = 'public'
  AND table_name LIKE 'v_%';
