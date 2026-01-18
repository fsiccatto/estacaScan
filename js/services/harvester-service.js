/**
 * Harvester Service
 * Gestión de cosechadores
 */

import { supabase } from '../supabase-client.js';

export class HarvesterService {

    /**
     * Obtener todos los cosechadores activos
     * @param {boolean} includeInactive - Incluir inactivos
     * @returns {Promise<Array>} Lista de cosechadores
     */
    static async getAll(includeInactive = false) {
        try {
            let query = supabase
                .from('harvesters')
                .select('*')
                .order('name');

            if (!includeInactive) {
                query = query.eq('is_active', true);
            }

            const { data, error } = await query;

            if (error) throw error;
            return data;

        } catch (error) {
            console.error('❌ Error al obtener cosechadores:', error);
            throw error;
        }
    }

    /**
     * Crear nuevo cosechador
     * @param {Object} harvester - Datos del cosechador
     * @returns {Promise<Object>} Cosechador creado
     */
    static async create(harvester) {
        try {
            const { data, error } = await supabase
                .from('harvesters')
                .insert([harvester])
                .select()
                .single();

            if (error) throw error;

            console.log('✅ Cosechador creado:', data.name);
            return data;

        } catch (error) {
            console.error('❌ Error al crear cosechador:', error);
            throw error;
        }
    }

    /**
     * Actualizar cosechador
     * @param {string} id - ID del cosechador
     * @param {Object} updates - Cambios a aplicar
     * @returns {Promise<Object>} Cosechador actualizado
     */
    static async update(id, updates) {
        try {
            const { data, error } = await supabase
                .from('harvesters')
                .update(updates)
                .eq('id', id)
                .select()
                .single();

            if (error) throw error;
            return data;

        } catch (error) {
            console.error('❌ Error al actualizar cosechador:', error);
            throw error;
        }
    }

    /**
     * Actualizar estadísticas de un cosechador
     * (Llamar después de guardar un análisis)
     * @param {string} harvesterId - ID del cosechador
     * @param {number} pricePerStake - Precio por estaca desde config
     * @returns {Promise<Object>} Cosechador actualizado
     */
    static async updateStats(harvesterId, pricePerStake) {
        try {
            // Sumar todas las estacas de este cosechador
            const { data: analyses, error } = await supabase
                .from('analysis')
                .select('total_confirmed')
                .eq('harvester_id', harvesterId);

            if (error) throw error;

            const totalStakes = analyses.reduce((sum, a) => sum + a.total_confirmed, 0);
            const totalEarnings = totalStakes * pricePerStake;

            // Actualizar stats
            return await this.update(harvesterId, {
                total_stakes: totalStakes,
                total_earnings: totalEarnings
            });

        } catch (error) {
            console.error('❌ Error al actualizar stats:', error);
            throw error;
        }
    }
}
