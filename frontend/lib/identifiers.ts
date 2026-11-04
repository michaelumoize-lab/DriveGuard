import crypto from "crypto";

/**
 * Canonical Business Identifier Prefixes for DriveGuard Core Entities
 *
 * Format: <PREFIX>-<6 NUMERIC DIGITS>
 * Examples:
 *  - Person:        PER-482910
 *  - Device:        DEV-918273
 *  - FaceEmbedding: EMB-123456
 *  - AccessLog:     ASL-456789
 *  - Alert:         ALT-987654
 */
export const IDENTIFIER_PREFIXES = {
  PERSON: "PER-",
  DEVICE: "DEV-",
  FACE_EMBEDDING: "EMB-",
  ACCESS_LOG: "ASL-",
  ALERT: "ALT-",
} as const;

export type IdentifierEntityType = keyof typeof IDENTIFIER_PREFIXES;
export type IdentifierPrefix = (typeof IDENTIFIER_PREFIXES)[IdentifierEntityType];

/**
 * Exact number of numeric digits after the 3-letter prefix and hyphen.
 */
export const SUFFIX_DIGIT_LENGTH = 6;

/**
 * Maximum exclusive bound for 6 numeric digits (000000 to 999999).
 */
const MAX_NUMERIC_SUFFIX_EXCLUSIVE = 1_000_000;

/**
 * Complete regular-expression patterns for each business entity code.
 */
export const PERSON_CODE_PATTERN = /^PER-[0-9]{6}$/;
export const DEVICE_CODE_PATTERN = /^DEV-[0-9]{6}$/;
export const EMBEDDING_CODE_PATTERN = /^EMB-[0-9]{6}$/;
export const ACCESS_CODE_PATTERN = /^ASL-[0-9]{6}$/;
export const ALERT_CODE_PATTERN = /^ALT-[0-9]{6}$/;

/**
 * Combined regular expression matching any valid canonical 5-entity business code.
 */
export const BUSINESS_ID_PATTERN = /^(PER|DEV|EMB|ASL|ALT)-[0-9]{6}$/;

const PREFIX_TO_PATTERN: Record<IdentifierPrefix, RegExp> = {
  [IDENTIFIER_PREFIXES.PERSON]: PERSON_CODE_PATTERN,
  [IDENTIFIER_PREFIXES.DEVICE]: DEVICE_CODE_PATTERN,
  [IDENTIFIER_PREFIXES.FACE_EMBEDDING]: EMBEDDING_CODE_PATTERN,
  [IDENTIFIER_PREFIXES.ACCESS_LOG]: ACCESS_CODE_PATTERN,
  [IDENTIFIER_PREFIXES.ALERT]: ALERT_CODE_PATTERN,
};

/**
 * Generates an unbiased, cryptographically secure 6-digit numeric string (000000 - 999999).
 * Uses Node.js `crypto.randomInt` which implements rejection sampling to eliminate modulo bias.
 */
function generateRandomSixDigits(): string {
  const num = crypto.randomInt(0, MAX_NUMERIC_SUFFIX_EXCLUSIVE);
  return num.toString().padStart(SUFFIX_DIGIT_LENGTH, "0");
}

/**
 * Core generator that creates a business identifier with the given prefix
 * and 6 cryptographically random numeric digits.
 */
export function generateBusinessIdentifier(prefix: IdentifierPrefix): string {
  return `${prefix}${generateRandomSixDigits()}`;
}

/**
 * Generates a unique business code for a Person record (e.g. `PER-482910`).
 */
export function generatePersonCode(): string {
  return generateBusinessIdentifier(IDENTIFIER_PREFIXES.PERSON);
}

/**
 * Generates a unique business code for a Device record (e.g. `DEV-918273`).
 */
export function generateDeviceCode(): string {
  return generateBusinessIdentifier(IDENTIFIER_PREFIXES.DEVICE);
}

/**
 * Generates a unique business code for a FaceEmbedding record (e.g. `EMB-123456`).
 */
export function generateFaceEmbeddingCode(): string {
  return generateBusinessIdentifier(IDENTIFIER_PREFIXES.FACE_EMBEDDING);
}

/**
 * Alias for `generateFaceEmbeddingCode()` (`EMB-123456`).
 */
export function generateEmbeddingCode(): string {
  return generateFaceEmbeddingCode();
}

/**
 * Generates a unique business code for an AccessLog record (e.g. `ASL-456789`).
 */
export function generateAccessLogCode(): string {
  return generateBusinessIdentifier(IDENTIFIER_PREFIXES.ACCESS_LOG);
}

/**
 * Alias for `generateAccessLogCode()` (`ASL-456789`).
 */
export function generateAccessCode(): string {
  return generateAccessLogCode();
}

/**
 * Generates a unique business code for an Alert record (e.g. `ALT-987654`).
 */
export function generateAlertCode(): string {
  return generateBusinessIdentifier(IDENTIFIER_PREFIXES.ALERT);
}

/**
 * Entity-specific validation helpers.
 */
export function isValidPersonCode(value: unknown): value is string {
  return typeof value === "string" && PERSON_CODE_PATTERN.test(value);
}

export function isValidDeviceCode(value: unknown): value is string {
  return typeof value === "string" && DEVICE_CODE_PATTERN.test(value);
}

export function isValidEmbeddingCode(value: unknown): value is string {
  return typeof value === "string" && EMBEDDING_CODE_PATTERN.test(value);
}

export function isValidAccessCode(value: unknown): value is string {
  return typeof value === "string" && ACCESS_CODE_PATTERN.test(value);
}

export function isValidAlertCode(value: unknown): value is string {
  return typeof value === "string" && ALERT_CODE_PATTERN.test(value);
}

/**
 * Validates whether a string is a well-formed DriveGuard business identifier.
 * Optionally checks that it matches a specific entity prefix.
 */
export function isValidBusinessIdentifier(
  value: unknown,
  expectedPrefix?: IdentifierPrefix
): value is string {
  if (typeof value !== "string" || value.length === 0) {
    return false;
  }

  if (expectedPrefix) {
    const pattern = PREFIX_TO_PATTERN[expectedPrefix];
    return pattern ? pattern.test(value) : false;
  }

  return BUSINESS_ID_PATTERN.test(value);
}

/**
 * Extracts the entity prefix from a valid business identifier, or returns null.
 */
export function getIdentifierPrefix(value: string): IdentifierPrefix | null {
  if (!isValidBusinessIdentifier(value)) {
    return null;
  }
  const prefix = value.slice(0, 4) as IdentifierPrefix;
  const validPrefixes = Object.values(IDENTIFIER_PREFIXES) as string[];
  return validPrefixes.includes(prefix) ? prefix : null;
}

/**
 * Checks whether an error is a database unique constraint violation (Prisma P2002 or PostgreSQL 23505).
 */
export function isUniqueConstraintError(error: unknown): boolean {
  if (!error || typeof error !== "object") {
    return false;
  }
  const errObj = error as { code?: unknown; message?: unknown };
  if (errObj.code === "P2002" || errObj.code === "23505") {
    return true;
  }
  if (
    typeof errObj.message === "string" &&
    (errObj.message.includes("Unique constraint failed") ||
      errObj.message.includes("duplicate key value violates unique constraint"))
  ) {
    return true;
  }
  return false;
}

/**
 * Collision-avoidance retry helper for persisting business identifiers.
 * Executes `generator()` and checks `existsCheck(candidate)` up to `maxRetries` times.
 * Does not swallow unrelated database errors thrown by `existsCheck`.
 */
export async function generateUniqueBusinessIdentifier(
  generator: () => string,
  existsCheck: (candidate: string) => Promise<boolean>,
  options: { maxRetries?: number; entityName?: string } = {}
): Promise<string> {
  const maxRetries = options.maxRetries ?? 5;
  const entityName = options.entityName ?? "Entity";

  if (!Number.isInteger(maxRetries) || maxRetries < 1) {
    throw new Error(`Invalid maxRetries (${maxRetries}) for ${entityName} identifier generation.`);
  }

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    const candidate = generator();
    const exists = await existsCheck(candidate);
    if (!exists) {
      return candidate;
    }
  }

  throw new Error(
    `Failed to generate unique business identifier for ${entityName} after ${maxRetries} attempts.`
  );
}
