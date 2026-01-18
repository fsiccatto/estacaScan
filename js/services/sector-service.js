/**
 * Sector Service
 * Gestión de sectores (ubicaciones físicas de la finca)
 */

import { supabase } from '../supabase-client.js';

export class SectorService {

    /**
     * Obtener todos los sectores activos
     * @param {boolean} includeInactive - Incluir inactivos
     * @returns {Promise<Array>} Lista de sectores
     */
    static async getAll(includeInactive = false) {
        try {
            let query = supabase
                .from('sectors')
                .select('*')
                .order('name');

            if (!includeInactive) {
                query = query.eq('is_active', true);
            }

            const { data, error } = await query;

            if (error) throw error;
            return data;

        } catch (error) {
            console.error('❌ Error al obtener sectores:', error);
            throw error;
        }
    }

    /**
     * Crear nuevo sector
     * @param {Object} sector - Datos del sector
     * @returns {Promise<Object>} Sector creado
     */
    static async create(sector) {
        try {
            const { data, error } = await supabase
                .from('sectors')
                .insert([sector])
                .select()
                .single();

            if (error) throw error;

            console.log('✅ Sector creado:', data.name);
            return data;

        } catch (error) {
            console.error('❌ Error al crear sector:', error);
            throw error;
        }
    }

    /**
     * Actualizar sector
     * @param {string} id - ID del sector
     * @param {Object} updates - Cambios a aplicar
     * @returns {Promise<Object>} Sector actualizado
     */
    static async update(id, updates) {
        try {
            const { data, error } = await supabase
                .from('sectors')
                .update(updates)
                .eq('id', id)
                .select()
                .single();

            if (error) throw error;
            return data;

        } catch (error) {
            console.error('❌ Error al actualizar sector:', error);
            throw error;
        }
    }
}
