-- ============================================================================
-- NeuroLens — Postgres bootstrap (runs once on first container init)
-- ----------------------------------------------------------------------------
-- This file is executed by the postgres:17-alpine image via the
-- /docker-entrypoint-initdb.d/ convention. It runs ONCE when the data
-- volume is empty — on subsequent restarts, the volume already has data
-- and this script is NOT re-executed.
--
-- Bootstrap order:
-- 1. Create the restricted role neurolens_writer
-- 2. Create the database neurolens owned by neurolens_writer
-- 3. The schema/tables are applied by 02-apply-schema.sql
-- ============================================================================

-- Read password from env var POSTGRES_NEUROLENS_WRITER_PASSWORD
-- (set in docker-compose.yml via env_file)
\set neurolens_password `echo "'$POSTGRES_NEUROLENS_WRITER_PASSWORD'"`

-- Create restricted role (NOT superuser; LOGIN-able with password)
CREATE ROLE neurolens_writer WITH LOGIN PASSWORD :neurolens_password;

-- Best practice: the user has just enough privileges to use the schema
ALTER ROLE neurolens_writer SET search_path TO neurolens, public;

-- ============================================================================
-- NOTE: We do NOT create the 'neurolens' database here because postgres:17
-- image already creates it from POSTGRES_DB env var (see docker-compose.yml).
-- We just grant the role usage on it.
-- ============================================================================

GRANT CONNECT ON DATABASE neurolens TO neurolens_writer;

-- Switch to neurolens DB for schema creation
\c neurolens

CREATE SCHEMA IF NOT EXISTS neurolens AUTHORIZATION neurolens_writer;
GRANT USAGE, CREATE ON SCHEMA neurolens TO neurolens_writer;

-- Default privileges for tables to be created later in this schema by the writer
ALTER DEFAULT PRIVILEGES IN SCHEMA neurolens
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO neurolens_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA neurolens
    GRANT USAGE, SELECT ON SEQUENCES TO neurolens_writer;
