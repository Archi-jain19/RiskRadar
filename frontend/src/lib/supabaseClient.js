/**
 * Supabase Client — Frontend
 * ===========================
 * Reads configuration from environment variables (Vite exposes VITE_* vars).
 *
 * Usage:
 *   import supabase from '../lib/supabaseClient';
 *   const { data, error } = await supabase.from('table').select('*');
 */

import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
    console.warn(
        '⚠️ Supabase environment variables are missing. ' +
        'Make sure VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY are set in frontend/.env'
    );
}

const supabase = createClient(supabaseUrl, supabaseAnonKey);

export default supabase;
