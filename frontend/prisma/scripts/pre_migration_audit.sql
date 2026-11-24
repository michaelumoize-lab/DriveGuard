-- DriveGuard Phase 0 Read-Only Pre-Migration Audit
-- Strictly read-only inspection queries (no DDL, no writes).

-- 1. Core table counts
SELECT 'user' AS tbl, COUNT(*)::int AS cnt FROM public."user"
UNION ALL SELECT 'account', COUNT(*)::int FROM public.account
UNION ALL SELECT 'session', COUNT(*)::int FROM public.session
UNION ALL SELECT 'verification', COUNT(*)::int FROM public.verification
UNION ALL SELECT 'user_profiles', COUNT(*)::int FROM public.user_profiles
UNION ALL SELECT 'driver_profiles', COUNT(*)::int FROM public.driver_profiles
UNION ALL SELECT 'driving_sessions', COUNT(*)::int FROM public.driving_sessions
UNION ALL SELECT 'drowsiness_alerts', COUNT(*)::int FROM public.drowsiness_alerts;

-- 2. Inspect driver_profiles rows & JSONB structure
SELECT
  dp.id,
  dp."userId",
  dp."calibrationMode",
  dp."isCalibrated",
  dp."calibratedAt",
  dp."sensitivityLevel",
  CASE WHEN jsonb_typeof(dp."zoneBaselines") = 'object'
       THEN ARRAY(SELECT jsonb_object_keys(dp."zoneBaselines") ORDER BY 1)
       ELSE NULL END AS zb_keys,
  (dp."zoneBaselines"->'CENTER' ? 'ear_median') AS center_has_snake_ear,
  (dp."zoneBaselines"->'CENTER' ? 'earMedian') AS center_has_camel_ear,
  (dp."zoneBaselines" ? '_poseMetadata') AS has_pose_meta,
  (dp."zoneBaselines" ? '_temporalBaseline') AS has_temporal_meta
FROM public.driver_profiles dp
ORDER BY dp."createdAt" ASC;

-- 3. Inspect user_profiles.sensitivityLevel distribution
SELECT "sensitivityLevel"::text AS sensitivity, COUNT(*)::int AS cnt
FROM public.user_profiles
GROUP BY 1
ORDER BY 1;

-- 4. Inspect PostgreSQL enums & foreign keys
SELECT t.typname AS enum_name, array_agg(e.enumlabel::text ORDER BY e.enumsortorder) AS labels
FROM pg_type t
JOIN pg_enum e ON t.oid = e.enumtypid
JOIN pg_namespace n ON n.oid = t.typnamespace
WHERE n.nspname = 'public'
GROUP BY t.typname
ORDER BY t.typname;

SELECT conrelid::regclass AS tbl, conname, convalidated, pg_get_constraintdef(oid) AS def
FROM pg_constraint
WHERE contype = 'f' AND connamespace = 'public'::regnamespace
ORDER BY 1, 2;
