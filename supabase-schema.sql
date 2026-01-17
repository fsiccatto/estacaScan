-- ===============================================
-- EstacaScan - Supabase Database Schema
-- NO STORAGE de imágenes - Solo metadata
-- ===============================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

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
  
  -- Configuración de pago
  price_per_stake DECIMAL(10,2) DEFAULT 0.00,
  
  -- Status
  is_active BOOLEAN DEFAULT true,
  
  -- Stats (calculadas desde analysis)
  total_stakes INT DEFAULT 0,
  total_earnings DECIMAL(10,2) DEFAULT 0.00,
  
  -- Metadata
  notes TEXT
);

-- Index for active harvesters
CREATE INDEX idx_harvesters_active ON harvesters(is_active) WHERE is_active = true;

-- ===============================================
-- Table: lots
-- Lotes/sectores del vivero
-- ===============================================
CREATE TABLE IF NOT EXISTS lots (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Info del lote
  name VARCHAR(100) NOT NULL,
  code VARCHAR(50) UNIQUE,
  location TEXT,
  
  -- Clasificación
  stake_type VARCHAR(100), -- ej: "Álamo", "Sauce", etc.
  client_name VARCHAR(255),
  
  -- Status
  is_active BOOLEAN DEFAULT true,
  
  -- Stats
  total_stakes INT DEFAULT 0,
  
  -- Metadata
  notes TEXT
);

-- Index for active lots
CREATE INDEX idx_lots_active ON lots(is_active) WHERE is_active = true;

-- ===============================================
-- Table: analysis
-- Historial de análisis (SIN imágenes)
-- ===============================================
CREATE TABLE IF NOT EXISTS analysis (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Relaciones
  harvester_id UUID REFERENCES harvesters(id) ON DELETE SET NULL,
  lot_id UUID REFERENCES lots(id) ON DELETE SET NULL,
  
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

-- Indexes for queries
CREATE INDEX idx_analysis_harvester ON analysis(harvester_id);
CREATE INDEX idx_analysis_lot ON analysis(lot_id);
CREATE INDEX idx_analysis_created ON analysis(created_at DESC);

-- ===============================================
-- Table: payments
-- Registro de pagos a cosechadores
-- ===============================================
CREATE TABLE IF NOT EXISTS payments (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Relación
  harvester_id UUID REFERENCES harvesters(id) ON DELETE RESTRICT NOT NULL,
  
  -- Período del pago
  period_start DATE NOT NULL,
  period_end DATE NOT NULL,
  
  -- Cálculo
  total_stakes INT NOT NULL,
  price_per_stake DECIMAL(10,2) NOT NULL,
  total_amount DECIMAL(10,2) NOT NULL,
  
  -- Estado
  status VARCHAR(20) DEFAULT 'pending', -- pending, paid, cancelled
  paid_at TIMESTAMPTZ,
  payment_method VARCHAR(50), -- cash, transfer, etc.
  
  -- Notas
  notes TEXT
);

-- Indexes
CREATE INDEX idx_payments_harvester ON payments(harvester_id);
CREATE INDEX idx_payments_status ON payments(status);
CREATE INDEX idx_payments_period ON payments(period_start, period_end);

-- ===============================================
-- Row Level Security (RLS)
-- Por ahora DISABLED para desarrollo
-- ===============================================
ALTER TABLE harvesters DISABLE ROW LEVEL SECURITY;
ALTER TABLE lots DISABLE ROW LEVEL SECURITY;
ALTER TABLE analysis DISABLE ROW LEVEL SECURITY;
ALTER TABLE payments DISABLE ROW LEVEL SECURITY;

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
CREATE TRIGGER update_harvesters_updated_at BEFORE UPDATE ON harvesters
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_lots_updated_at BEFORE UPDATE ON lots
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_payments_updated_at BEFORE UPDATE ON payments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ===============================================
-- Sample Data (opcional)
-- ===============================================

-- Insertar cosechadores de ejemplo
INSERT INTO harvesters (name, dni, price_per_stake) VALUES
  ('Juan Pérez', '12345678', 0.50),
  ('María González', '87654321', 0.45)
ON CONFLICT (dni) DO NOTHING;

-- Insertar lotes de ejemplo  
INSERT INTO lots (name, code, stake_type) VALUES
  ('Lote A - Norte', 'LT-A', 'Álamo'),
  ('Lote B - Sur', 'LT-B', 'Sauce')
ON CONFLICT (code) DO NOTHING;

-- ===============================================
-- Verificación
-- ===============================================

-- Query para verificar tablas creadas
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name IN ('harvesters', 'lots', 'analysis', 'payments');
