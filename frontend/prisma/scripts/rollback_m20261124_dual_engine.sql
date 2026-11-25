-- DriveGuard Phase 0 Non-Truncating, FK-Preserving Rollback: m20261124_dual_engine
-- Preserves original rows in place and archives any post-migration data in JSONB retention tables.

BEGIN;

CREATE TABLE public.rollback_retained_m20261124_driver_profiles AS
SELECT id, CURRENT_TIMESTAMP AS archived_at, to_jsonb(t) AS row_snapshot FROM public.driver_profiles t;

CREATE TABLE public.rollback_retained_m20261124_profile_history AS
SELECT id, CURRENT_TIMESTAMP AS archived_at, to_jsonb(t) AS row_snapshot FROM public.driver_profile_history t;

CREATE TABLE public.rollback_retained_m20261124_user_relaxed AS
SELECT id, CURRENT_TIMESTAMP AS archived_at, to_jsonb(t) AS row_snapshot
FROM public.user_profiles t
WHERE "sensitivityLevel"::text = 'RELAXED';

-- Remove only newly added secondary engine rows for users with multiple rows (original row stays in place)
DELETE FROM public.driver_profiles dp
WHERE dp.id NOT IN (SELECT id FROM public.backup_m20261124_driver_profiles)
  AND EXISTS (
    SELECT 1 FROM public.driver_profiles other
    WHERE other."userId" = dp."userId" AND other.id <> dp.id
  );

DROP TABLE public.driver_profile_history;
DROP INDEX IF EXISTS public."driver_profiles_userId_engineType_key";
DROP INDEX IF EXISTS public."driver_profiles_profileUuid_key";

ALTER TABLE public.driver_profiles
  DROP COLUMN "engineType",
  DROP COLUMN "profileVersion",
  DROP COLUMN "profileUuid",
  DROP COLUMN "contentHash",
  DROP COLUMN "neutralRotMat",
  DROP COLUMN "calibrationIsMirrored";

CREATE UNIQUE INDEX "driver_profiles_userId_key" ON public.driver_profiles USING btree ("userId");

DROP INDEX IF EXISTS public."driving_sessions_clientSessionId_key";
DROP INDEX IF EXISTS public."driving_sessions_userId_engineType_idx";
ALTER TABLE public.driving_sessions
  DROP COLUMN "engineType",
  DROP COLUMN "clientSessionId",
  DROP COLUMN "status",
  DROP COLUMN "finalizeToken";

DROP INDEX IF EXISTS public."drowsiness_alerts_eventId_key";
ALTER TABLE public.drowsiness_alerts
  DROP COLUMN "eventId";

DROP TYPE public."SessionStatus";
DROP TYPE public."EngineType";

CREATE TYPE public."SensitivityLevel_orig" AS ENUM ('STANDARD', 'HIGH', 'HIGHWAY');
ALTER TABLE public.user_profiles
  ALTER COLUMN "sensitivityLevel" DROP DEFAULT,
  ALTER COLUMN "sensitivityLevel" TYPE public."SensitivityLevel_orig"
    USING (
      CASE WHEN "sensitivityLevel"::text = 'RELAXED' THEN 'STANDARD'
           ELSE "sensitivityLevel"::text
      END::public."SensitivityLevel_orig"
    ),
  ALTER COLUMN "sensitivityLevel" SET DEFAULT 'STANDARD'::public."SensitivityLevel_orig";
DROP TYPE public."SensitivityLevel";
ALTER TYPE public."SensitivityLevel_orig" RENAME TO "SensitivityLevel";

DO $$
BEGIN
  IF (SELECT COUNT(*) FROM public.driver_profiles WHERE id IN (SELECT id FROM public.backup_m20261124_driver_profiles))
     <> (SELECT COUNT(*) FROM public.backup_m20261124_driver_profiles) THEN
    RAISE EXCEPTION 'Rollback failed: pre-migration driver_profiles rows missing';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'driver_profiles_userId_fkey' AND convalidated) THEN
    RAISE EXCEPTION 'Rollback failed: driver_profiles_userId_fkey missing or invalid';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'driver_profiles_userId_key') THEN
    RAISE EXCEPTION 'Rollback failed: driver_profiles_userId_key missing';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'drowsiness_alerts_sessionId_fkey' AND convalidated) THEN
    RAISE EXCEPTION 'Rollback failed: drowsiness_alerts_sessionId_fkey missing or invalid';
  END IF;
  IF (SELECT array_agg(enumlabel::text ORDER BY enumsortorder) FROM pg_enum WHERE enumtypid = 'public."SensitivityLevel"'::regtype)
     <> ARRAY['STANDARD', 'HIGH', 'HIGHWAY']::text[] THEN
    RAISE EXCEPTION 'Rollback failed: SensitivityLevel enum not restored';
  END IF;
END $$;

COMMIT;
