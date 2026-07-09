-- SupplyChainOptix Database Schema
-- PostgreSQL 14+ required for certain JSONB features
-- This schema supports real-time supply chain analytics, ML predictions, and geopolitical event tracking

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Set timezone to UTC for consistency
SET timezone = 'UTC';

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- Suppliers master data
CREATE TABLE suppliers (
    supplier_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    supplier_code VARCHAR(50) UNIQUE NOT NULL,
    supplier_name VARCHAR(255) NOT NULL,
    country_code VARCHAR(3) NOT NULL,
    region VARCHAR(100),
    tier INTEGER CHECK (tier BETWEEN 1 AND 3), -- Supply chain tier level
    reliability_score DECIMAL(3,2) CHECK (reliability_score BETWEEN 0 AND 1),
    contact_info JSONB,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_suppliers_country ON suppliers(country_code);
CREATE INDEX idx_suppliers_tier ON suppliers(tier);
CREATE INDEX idx_suppliers_active ON suppliers(active) WHERE active = TRUE;

-- Products and SKUs
CREATE TABLE products (
    product_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sku VARCHAR(100) UNIQUE NOT NULL,
    product_name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    subcategory VARCHAR(100),
    unit_cost DECIMAL(12,2),
    lead_time_days INTEGER,
    minimum_order_quantity INTEGER,
    critical_stock_flag BOOLEAN DEFAULT FALSE,
    attributes JSONB, -- Flexible attributes like weight, dimensions, etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_critical ON products(critical_stock_flag) WHERE critical_stock_flag = TRUE;
CREATE INDEX idx_products_sku_trgm ON products USING gin(sku gin_trgm_ops);

-- Supplier-Product relationships
CREATE TABLE supplier_products (
    supplier_product_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    supplier_id UUID NOT NULL REFERENCES suppliers(supplier_id) ON DELETE CASCADE,
    product_id UUID NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    unit_cost DECIMAL(12,2),
    lead_time_days INTEGER,
    minimum_order_quantity INTEGER,
    preferred_supplier BOOLEAN DEFAULT FALSE,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(supplier_id, product_id)
);

CREATE INDEX idx_supplier_products_supplier ON supplier_products(supplier_id);
CREATE INDEX idx_supplier_products_product ON supplier_products(product_id);

-- Purchase orders
CREATE TABLE purchase_orders (
    po_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    po_number VARCHAR(50) UNIQUE NOT NULL,
    supplier_id UUID NOT NULL REFERENCES suppliers(supplier_id),
    order_date DATE NOT NULL,
    expected_delivery_date DATE NOT NULL,
    actual_delivery_date DATE,
    status VARCHAR(50) NOT NULL CHECK (status IN ('pending', 'confirmed', 'in_transit', 'delivered', 'delayed', 'cancelled')),
    total_amount DECIMAL(15,2),
    currency VARCHAR(3) DEFAULT 'USD',
    shipping_method VARCHAR(100),
    tracking_number VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_po_supplier ON purchase_orders(supplier_id);
CREATE INDEX idx_po_status ON purchase_orders(status);
CREATE INDEX idx_po_expected_delivery ON purchase_orders(expected_delivery_date);
CREATE INDEX idx_po_order_date ON purchase_orders(order_date DESC);

-- Purchase order line items
CREATE TABLE po_line_items (
    po_line_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    po_id UUID NOT NULL REFERENCES purchase_orders(po_id) ON DELETE CASCADE,
    product_id UUID NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price DECIMAL(12,2) NOT NULL,
    line_total DECIMAL(15,2) GENERATED ALWAYS AS (quantity * unit_price) STORED,
    received_quantity INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_po_line_po ON po_line_items(po_id);
CREATE INDEX idx_po_line_product ON po_line_items(product_id);

-- Inventory levels
CREATE TABLE inventory (
    inventory_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_id UUID NOT NULL REFERENCES products(product_id),
    warehouse_code VARCHAR(50) NOT NULL,
    quantity_on_hand INTEGER NOT NULL DEFAULT 0,
    quantity_reserved INTEGER NOT NULL DEFAULT 0,
    quantity_available INTEGER GENERATED ALWAYS AS (quantity_on_hand - quantity_reserved) STORED,
    reorder_point INTEGER,
    safety_stock INTEGER,
    last_stock_check TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(product_id, warehouse_code)
);

CREATE INDEX idx_inventory_product ON inventory(product_id);
CREATE INDEX idx_inventory_warehouse ON inventory(warehouse_code);
CREATE INDEX idx_inventory_low_stock ON inventory(quantity_available) WHERE quantity_available < reorder_point;

-- Historical inventory snapshots for trend analysis
CREATE TABLE inventory_snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_id UUID NOT NULL REFERENCES products(product_id),
    warehouse_code VARCHAR(50) NOT NULL,
    snapshot_date DATE NOT NULL,
    quantity_on_hand INTEGER NOT NULL,
    quantity_reserved INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(product_id, warehouse_code, snapshot_date)
);

CREATE INDEX idx_inv_snapshots_product_date ON inventory_snapshots(product_id, snapshot_date DESC);
CREATE INDEX idx_inv_snapshots_date ON inventory_snapshots(snapshot_date DESC);

-- =============================================================================
-- DISRUPTION & EVENT TRACKING
-- =============================================================================

-- Geopolitical and external events
CREATE TABLE external_events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL, -- e.g., 'geopolitical', 'natural_disaster', 'port_congestion', 'labor_strike'
    event_name VARCHAR(255) NOT NULL,
    description TEXT,
    affected_countries VARCHAR(3)[], -- Array of ISO country codes
    affected_regions VARCHAR(100)[],
    severity VARCHAR(20) CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    start_date DATE NOT NULL,
    end_date DATE,
    source VARCHAR(255), -- Data source
    source_url TEXT,
    metadata JSONB, -- Additional flexible data
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_events_type ON external_events(event_type);
CREATE INDEX idx_events_severity ON external_events(severity);
CREATE INDEX idx_events_start_date ON external_events(start_date DESC);
CREATE INDEX idx_events_countries ON external_events USING gin(affected_countries);

-- Actual supply chain disruptions
CREATE TABLE disruptions (
    disruption_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    supplier_id UUID REFERENCES suppliers(supplier_id),
    product_id UUID REFERENCES products(product_id),
    event_id UUID REFERENCES external_events(event_id),
    disruption_type VARCHAR(100) NOT NULL, -- e.g., 'delay', 'quality_issue', 'shortage', 'price_increase'
    severity VARCHAR(20) CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    detected_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE,
    impact_description TEXT,
    estimated_delay_days INTEGER,
    actual_delay_days INTEGER,
    affected_po_ids UUID[],
    mitigation_actions TEXT,
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'monitoring', 'resolved', 'mitigated')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_disruptions_supplier ON disruptions(supplier_id);
CREATE INDEX idx_disruptions_product ON disruptions(product_id);
CREATE INDEX idx_disruptions_event ON disruptions(event_id);
CREATE INDEX idx_disruptions_status ON disruptions(status);
CREATE INDEX idx_disruptions_detected ON disruptions(detected_at DESC);

-- =============================================================================
-- ML PREDICTIONS & ANALYTICS
-- =============================================================================

-- ML model metadata and versioning
CREATE TABLE ml_models (
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- e.g., 'prophet', 'isolation_forest', 'random_forest'
    model_version VARCHAR(50) NOT NULL,
    purpose VARCHAR(255), -- e.g., 'demand_forecasting', 'anomaly_detection', 'delay_prediction'
    hyperparameters JSONB,
    training_date TIMESTAMP WITH TIME ZONE NOT NULL,
    training_metrics JSONB, -- Accuracy, precision, recall, etc.
    active BOOLEAN DEFAULT TRUE,
    model_path VARCHAR(500), -- File path or S3 URI
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, model_version)
);

CREATE INDEX idx_ml_models_active ON ml_models(active, model_name);
CREATE INDEX idx_ml_models_type ON ml_models(model_type);

-- Delay predictions
CREATE TABLE delay_predictions (
    prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES ml_models(model_id),
    po_id UUID REFERENCES purchase_orders(po_id),
    supplier_id UUID REFERENCES suppliers(supplier_id),
    product_id UUID REFERENCES products(product_id),
    prediction_date TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    predicted_delay_days INTEGER NOT NULL,
    confidence_score DECIMAL(3,2) CHECK (confidence_score BETWEEN 0 AND 1),
    contributing_factors JSONB, -- Feature importance from model
    risk_level VARCHAR(20) CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    forecast_window_days INTEGER, -- How many days ahead (7-14)
    actual_delay_days INTEGER, -- Filled after actual event
    prediction_accuracy DECIMAL(5,2), -- Calculated post-event
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_delay_pred_po ON delay_predictions(po_id);
CREATE INDEX idx_delay_pred_supplier ON delay_predictions(supplier_id);
CREATE INDEX idx_delay_pred_date ON delay_predictions(prediction_date DESC);
CREATE INDEX idx_delay_pred_risk ON delay_predictions(risk_level);

-- Demand forecasts
CREATE TABLE demand_forecasts (
    forecast_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES ml_models(model_id),