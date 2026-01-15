import process from "node:process";
import { createClient } from "@supabase/supabase-js";

function resolveSupabaseConfig(overrides = {}) {
  const url =
    overrides.url ||
    process.env.SUPABASE_URL ||
    process.env.NEXT_PUBLIC_SUPABASE_URL ||
    "";
  const key =
    overrides.key ||
    process.env.SUPABASE_SERVICE_ROLE_KEY ||
    process.env.SUPABASE_ANON_KEY ||
    "";

  return {
    url: url.trim(),
    key: key.trim(),
  };
}

export function createSupabase(overrides = {}) {
  const { url, key } = resolveSupabaseConfig(overrides);

  if (!url) {
    throw new Error("Supabase URL is required.");
  }

  if (!key) {
    throw new Error("Supabase key is required.");
  }

  return createClient(url, key, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
}

export function hasSupabaseConfig(overrides = {}) {
  try {
    const { url, key } = resolveSupabaseConfig(overrides);
    return Boolean(url && key);
  } catch (error) {
    return false;
  }
}
