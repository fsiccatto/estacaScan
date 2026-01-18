/**
 * Batch Service
 * Gestión de lotes de estacas (grupos por variedad)
 */

import { supabase } from '../supabase-client.js';

export class BatchService {

    /**
     * Obtener todos los lotes activos
     * @param {boolean} includeInactive - Incluir inactivos
     * @returns {Promise<Array>} Lista de lotes
     */
    static async getAll(includeInactive = false) {
        try {
            let query = supabase
                .from('batches')
                .select('*')
                .order('created_at', { ascending: false });

            if (!includeInactive) {
                query = query.eq('is_active', true);
            }

            const { data, error } = await query;

            if (error) throw error;
            return data;

        } catch (error) {
            console.error('❌ Error al obtener lotes:', error);
            throw error;
        }
    }

    /**
     * Obtener lotes por estado
     * @param {string} status - Estado del lote
     * @returns {Promise<Array>} Lista de lotes
     */
    static async getByStatus(status) {
        try {
            const { data, error } = await supabase
                .from('batches')
                .select('*')
                .eq('status', status)
                .eq('is_active', true)
                .order('created_at', { ascending: false });

            if (error) throw error;
            return data;

        } catch (error) {
            console.error('❌ Error al obtener lotes por estado:', error);
            throw error;
        }
    }

    /**
     * Crear nuevo lote
     * @param {Object} batch - Datos del lote
     * @returns {Promise<Object>} Lote creado
     */
    static async create(batch) {
        try {
            const { data, error } = await supabase
                .from('batches')
                .insert([batch])
                .select()
                .single();

            if (error) throw error;

            console.log('✅ Lote creado:', data.code);
            return data;

        } catch (error) {
            console.error('❌ Error al crear lote:', error);
            throw error;
        }
    }

    /**
     * Actualizar lote
     * @param {string} id - ID del lote
     * @param {Object} updates - Cambios a aplicar
     * @returns {Promise<Object>} Lote actualizado
     */
    static async update(id, updates) {
        try {
            const { data, error } = await supabase
                .from('batches')
                .update(updates)
                .eq('id', id)
                .select()
                .single();

            if (error) throw error;
            return data;

        } catch (error) {
            console.error('❌ Error al actualizar lote:', error);
            throw error;
        }
    }

    /**
     * Cambiar estado de un lote
     * @param {string} id - ID del lote
     * @param {string} newStatus - Nuevo estado
     * @returns {Promise<Object>} Lote actualizado
     */
    static async changeStatus(id, newStatus) {
        return await this.update(id, { status: newStatus });
    }

    /**
     * Actualizar total de estacas de un lote
     * (Llamar después de guardar análisis de este lote)
     * @param {string} batchId - ID del lote
     * @returns {Promise<Object>} Lote actualizado
     */
    static async updateTotal(batchId) {
        try {
            // Sumar todas las estacas de este lote
            const { data: analyses, error } = await supabase
                .from('analysis')
                .select('total_confirmed')
                .eq('batch_id', batchId);

            if (error) throw error;

            const totalStakes = analyses.reduce((sum, a) => sum + a.total_confirmed, 0);

            // Actualizar total
            return await this.update(batchId, { total_stakes: totalStakes });

        } catch (error) {
            console.error('❌ Error al actualizar total del lote:', error);
            throw error;
        }
    }
}
