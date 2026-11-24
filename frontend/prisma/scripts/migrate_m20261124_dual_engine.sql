-- DriveGuard Phase 0 Forward Migration: m20261124_dual_engine
-- Additive, checksum-verified, evidence-gated dual-engine schema migration.
-- Verified on disposable Neon branch rehearsal prior to production execution.

BEGIN;

-- 1. Migration-specific backup snapshots (fails loudly if already present; enum stored as text)
CREATE TABLE public.backup_m20261124_driver_profiles AS TABLE public.driver_profiles;
CREATE TABLE public.backup_m20261124_user_profiles AS
SELECT id, "userId", "onboardingCompleted", "onboardingStep", "completedSteps",
       "phoneNumber", "licenseNumber", "vehicleMake", "vehicleModel", "vehicleYear",
       "licensePlate", "emergencyContactName", "emergencyContactPhone",
       "sensitivityLevel"::text AS "sensitivityLevel",
       "audioAlertsEnabled", "mirrorSuppression", "wearsGlasses", "createdAt", "updatedAt"
FROM public.user_profiles;
CREATE TABLE public.backup_m20261124_driving_sessions AS TABLE public.driving_sessions;
CREATE TABLE public.backup_m20261124_drowsiness_alerts AS TABLE public.drowsiness_alerts;

DO $$
DECLARE
  v_src_cnt int; v_bak_cnt int;
  v_src_md5 text; v_bak_md5 text;
BEGIN
  SELECT COUNT(*), COALESCE(md5(string_agg(to_jsonb(t)::text, '|' ORDER BY id)), '') INTO v_src_cnt, v_src_md5 FROM public.driver_profiles t;
  SELECT COUNT(*), COALESCE(md5(string_agg(to_jsonb(t)::text, '|' ORDER BY id)), '') INTO v_bak_cnt, v_bak_md5 FROM public.backup_m20261124_driver_profiles t;
  IF v_src_cnt <> v_bak_cnt OR v_src_md5 <> v_bak_md5 THEN
    RAISE EXCEPTION 'Backup checksum mismatch for driver_profiles (% vs %)', v_src_cnt, v_bak_cnt;
  END IF;

  SELECT COUNT(*), COALESCE(md5(string_agg(to_jsonb(t)::text, '|' ORDER BY id)), '') INTO v_src_cnt, v_src_md5 FROM public.user_profiles t;
  SELECT COUNT(*), COALESCE(md5(string_agg(to_jsonb(t)::text, '|' ORDER BY id)), '') INTO v_bak_cnt, v_bak_md5 FROM public.backup_m20261124_user_profiles t;
  IF v_src_cnt <> v_bak_cnt OR v_src_md5 <> v_bak_md5 THEN
    RAISE EXCEPTION 'Backup checksum mismatch for user_profiles';
  END IF;

  SELECT COUNT(*), COALESCE(md5(string_agg(to_jsonb(t)::text, '|' ORDER BY id)), '') INTO v_src_cnt, v_src_md5 FROM public.driving_sessions t;
  SELECT COUNT(*), COALESCE(md5(string_agg(to_jsonb(t)::text, '|' ORDER BY id)), '') INTO v_bak_cnt, v_bak_md5 FROM public.backup_m20261124_driving_sessions t;
  IF v_src_cnt <> v_bak_cnt OR v_src_md5 <> v_bak_md5 THEN
    RAISE EXCEPTION 'Backup checksum mismatch for driving_sessions';
  END IF;

  SELECT COUNT(*), COALESCE(md5(string_agg(to_jsonb(t)::text, '|' ORDER BY id)), '') INTO v_src_cnt, v_src_md5 FROM public.drowsiness_alerts t;
  SELECT COUNT(*), COALESCE(md5(string_agg(to_jsonb(t)::text, '|' ORDER BY id)), '') INTO v_bak_cnt, v_bak_md5 FROM public.backup_m20261124_drowsiness_alerts t;
  IF v_src_cnt <> v_bak_cnt OR v_src_md5 <> v_bak_md5 THEN
    RAISE EXCEPTION 'Backup checksum mismatch for drowsiness_alerts';
  END IF;
END $$;

-- 2. New enums
CREATE TYPE public."EngineType" AS ENUM ('BROWSER', 'PYTHON');
CREATE TYPE public."SessionStatus" AS ENUM ('ACTIVE', 'COMPLETED');

-- 3. Evidence-gated profile classification staging (halts if any LEGACY_UNKNOWN row exists)
CREATE TABLE public.legacy_profile_classification_staging_m20261124 (
  "profileId"       TEXT PRIMARY KEY,
  "userId"          TEXT NOT NULL,
  "proposedEngine"  TEXT NOT NULL,
  "evidenceSummary" TEXT NOT NULL,
  "inspectedAt"     TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO public.legacy_profile_classification_staging_m20261124
  ("profileId", "userId", "proposedEngine", "evidenceSummary")
SELECT
  dp.id,
  dp."userId",
  CASE
    WHEN dp."isCalibrated" = true
     AND dp."calibrationMode" IN ('4_ZONE', 'FOUR_ZONE', 'THREE_ZONE', 'CENTER_ONLY')
     AND (dp."zoneBaselines"->'CENTER' ? 'ear_median')
     AND NOT (dp."zoneBaselines"->'CENTER' ? 'earMedian')
     AND dp."calibratedAt" < TIMESTAMP '2026-11-22 00:00:00'
      THEN 'PYTHON'
    WHEN dp."isCalibrated" = true
     AND (dp."zoneBaselines"->'CENTER' ? 'earMedian')
     AND NOT (dp."zoneBaselines"->'CENTER' ? 'ear_median')
      THEN 'BROWSER'
    ELSE 'LEGACY_UNKNOWN'
  END,
  CASE
    WHEN dp."isCalibrated" = true
     AND (dp."zoneBaselines"->'CENTER' ? 'ear_median')
     AND dp."calibratedAt" < TIMESTAMP '2026-11-22 00:00:00'
      THEN 'Verified snake_case 4_ZONE Python calibration predating Browser AI'
    WHEN dp."isCalibrated" = true
     AND (dp."zoneBaselines"->'CENTER' ? 'earMedian')
      THEN 'Verified camelCase Browser calibration'
    ELSE 'Ambiguous or uncalibrated legacy record — manual resolution required'
  END
FROM public.driver_profiles dp;

DO $$
DECLARE v_unknown_cnt int;
BEGIN
  SELECT COUNT(*) INTO v_unknown_cnt
  FROM public.legacy_profile_classification_staging_m20261124
  WHERE "proposedEngine" = 'LEGACY_UNKNOWN';
  IF v_unknown_cnt > 0 THEN
    RAISE EXCEPTION 'Migration halted: % ambiguous driver_profiles row(s) classified as LEGACY_UNKNOWN.', v_unknown_cnt;
  END IF;
END $$;

-- 4. Additive driver_profiles updates (preserves legacy zoneBaselines JSONB intact)
UPDATE public.driver_profiles
SET
  "blinkFreqMedianPerMin" = COALESCE("blinkFreqMedianPerMin", 16.0),
  "blinkFreqMadPerMin"    = COALESCE("blinkFreqMadPerMin", 4.5),
  "baselinePerclosPct"    = COALESCE("baselinePerclosPct", 3.5),
  "blinkAmplitudeMedian"  = COALESCE("blinkAmplitudeMedian", 0.14),
  "temporalBlinksSampled" = COALESCE("temporalBlinksSampled", 0),
  "temporalWindowSec"     = COALESCE("temporalWindowSec", 120.0),
  "sensitivityLevel"      = COALESCE("sensitivityLevel", 'STANDARD');

ALTER TABLE public.driver_profiles
  ALTER COLUMN "blinkFreqMedianPerMin" SET NOT NULL,
  ALTER COLUMN "blinkFreqMadPerMin"    SET NOT NULL,
  ALTER COLUMN "baselinePerclosPct"    SET NOT NULL,
  ALTER COLUMN "blinkAmplitudeMedian"  SET NOT NULL,
  ALTER COLUMN "temporalBlinksSampled" SET NOT NULL,
  ALTER COLUMN "temporalWindowSec"     SET NOT NULL,
  ALTER COLUMN "sensitivityLevel"      SET NOT NULL,
  ADD COLUMN "engineType"            public."EngineType",
  ADD COLUMN "profileVersion"        INTEGER NOT NULL DEFAULT 1,
  ADD COLUMN "profileUuid"           TEXT,
  ADD COLUMN "contentHash"           TEXT,
  ADD COLUMN "neutralRotMat"         JSONB,
  ADD COLUMN "calibrationIsMirrored" BOOLEAN;

UPDATE public.driver_profiles dp
SET
  "engineType"            = st."proposedEngine"::public."EngineType",
  "profileUuid"           = gen_random_uuid()::text,
  "neutralRotMat"         = CASE
                              WHEN dp."zoneBaselines" ? '_poseMetadata'
                               AND dp."zoneBaselines"->'_poseMetadata' ? 'neutral_rot_mat'
                               AND dp."zoneBaselines"->'_poseMetadata'->>'neutral_rot_mat' <> 'null'
                              THEN dp."zoneBaselines"->'_poseMetadata'->'neutral_rot_mat'
                              ELSE NULL
                            END,
  "calibrationIsMirrored" = CASE
                              WHEN dp."zoneBaselines" ? '_poseMetadata'
                               AND jsonb_typeof(dp."zoneBaselines"->'_poseMetadata'->'calibration_is_mirrored') = 'boolean'
                              THEN (dp."zoneBaselines"->'_poseMetadata'->>'calibration_is_mirrored')::boolean
                              ELSE NULL
                            END
FROM public.legacy_profile_classification_staging_m20261124 st
WHERE dp.id = st."profileId";

ALTER TABLE public.driver_profiles
  ALTER COLUMN "engineType"  SET NOT NULL,
  ALTER COLUMN "profileUuid" SET NOT NULL;

CREATE UNIQUE INDEX "driver_profiles_userId_engineType_key" ON public.driver_profiles USING btree ("userId", "engineType");
CREATE UNIQUE INDEX "driver_profiles_profileUuid_key" ON public.driver_profiles USING btree ("profileUuid");
DROP INDEX public."driver_profiles_userId_key";

-- 5. driver_profile_history table + baseline snapshot
CREATE TABLE public.driver_profile_history (
  id                   TEXT PRIMARY KEY,
  "driverProfileId"    TEXT NOT NULL,
  "userId"             TEXT NOT NULL,
  "engineType"         public."EngineType" NOT NULL,
  "profileVersion"     INTEGER NOT NULL,
  "profileUuid"        TEXT NOT NULL,
  "contentHash"        TEXT,
  "snapshot"           JSONB NOT NULL,
  "clientCalibratedAt" TIMESTAMP(3),
  "serverRecordedAt"   TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "archiveReason"      TEXT NOT NULL,
  CONSTRAINT "driver_profile_history_driverProfileId_fkey"
    FOREIGN KEY ("driverProfileId") REFERENCES public.driver_profiles(id)
    ON UPDATE CASCADE ON DELETE CASCADE,
  CONSTRAINT "driver_profile_history_userId_fkey"
    FOREIGN KEY ("userId") REFERENCES public."user"(id)
    ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE INDEX "driver_profile_history_userId_engineType_idx"
  ON public.driver_profile_history USING btree ("userId", "engineType", "serverRecordedAt" DESC);
CREATE INDEX "driver_profile_history_driverProfileId_idx"
  ON public.driver_profile_history USING btree ("driverProfileId");

INSERT INTO public.driver_profile_history
  (id, "driverProfileId", "userId", "engineType", "profileVersion", "profileUuid", "contentHash", "snapshot", "clientCalibratedAt", "archiveReason")
SELECT
  'dph_' || replace(gen_random_uuid()::text, '-', ''),
  dp.id,
  dp."userId",
  dp."engineType",
  dp."profileVersion",
  dp."profileUuid",
  dp."contentHash",
  to_jsonb(dp),
  dp."calibratedAt",
  'MIGRATION_BASELINE'
FROM public.driver_profiles dp;

-- 6. Additive updates to driving_sessions (historical engineType stays NULL) & drowsiness_alerts
ALTER TABLE public.driving_sessions
  ADD COLUMN "engineType"      public."EngineType" NULL,
  ADD COLUMN "clientSessionId" TEXT NULL,
  ADD COLUMN "status"          public."SessionStatus" NOT NULL DEFAULT 'ACTIVE',
  ADD COLUMN "finalizeToken"   TEXT NULL;

UPDATE public.driving_sessions SET "status" = 'COMPLETED' WHERE "endedAt" IS NOT NULL;

CREATE UNIQUE INDEX "driving_sessions_clientSessionId_key" ON public.driving_sessions USING btree ("clientSessionId");
CREATE INDEX "driving_sessions_userId_engineType_idx" ON public.driving_sessions USING btree ("userId", "engineType");

UPDATE public.drowsiness_alerts SET reasons = ARRAY[]::text[] WHERE reasons IS NULL;
ALTER TABLE public.drowsiness_alerts
  ALTER COLUMN reasons SET NOT NULL,
  ALTER COLUMN reasons SET DEFAULT ARRAY[]::text[],
  ADD COLUMN "eventId" TEXT NULL;

CREATE UNIQUE INDEX "drowsiness_alerts_eventId_key" ON public.drowsiness_alerts USING btree ("eventId");

-- 7. SensitivityLevel enum update (STANDARD | HIGH | HIGHWAY -> RELAXED | STANDARD | HIGH)
UPDATE public.user_profiles SET "completedSteps" = ARRAY[]::text[] WHERE "completedSteps" IS NULL;
ALTER TABLE public.user_profiles ALTER COLUMN "completedSteps" SET NOT NULL;

DO $$
DECLARE v_hw_cnt int;
BEGIN
  SELECT COUNT(*) INTO v_hw_cnt FROM public.user_profiles WHERE "sensitivityLevel"::text = 'HIGHWAY';
  IF v_hw_cnt > 0 THEN
    RAISE EXCEPTION 'Found % user_profiles row(s) with HIGHWAY; review mapping before enum swap', v_hw_cnt;
  END IF;
END $$;

CREATE TYPE public."SensitivityLevel_v2" AS ENUM ('RELAXED', 'STANDARD', 'HIGH');
ALTER TABLE public.user_profiles
  ALTER COLUMN "sensitivityLevel" DROP DEFAULT,
  ALTER COLUMN "sensitivityLevel" TYPE public."SensitivityLevel_v2"
    USING ("sensitivityLevel"::text::public."SensitivityLevel_v2"),
  ALTER COLUMN "sensitivityLevel" SET DEFAULT 'STANDARD'::public."SensitivityLevel_v2";
DROP TYPE public."SensitivityLevel";
ALTER TYPE public."SensitivityLevel_v2" RENAME TO "SensitivityLevel";

COMMIT;
