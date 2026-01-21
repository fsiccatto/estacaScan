/**
 * Supabase Client Configuration
 * Singleton para interactuar con la base de datos
 */

import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/+esm';

const SUPABASE_URL = 'https://vnjxwvyifhjicmgngtun.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZuanh3dnlpZmhqaWNtZ25ndHVuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njc3NDE5NDcsImV4cCI6MjA4MzMxNzk0N30.zzWKBCCJh4hkxpkLTX1ZZDlXLxM6CnbEzRZiYyr6Fw8';

// Initialize client
const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// Test connection
async function testConnection() {
    try {
        const { data, error } = await supabase.from('config').select('*').limit(1);
        if (error) throw error;
        console.log('✅ Supabase connected successfully');
        return true;
    } catch (error) {
        console.error('❌ Supabase connection error:', error);
        return false;
    }
}

// Export for use in other modules
export { supabase, testConnection };
