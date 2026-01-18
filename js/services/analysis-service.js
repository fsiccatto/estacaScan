/**
 * Analysis Service
 * Manejo de análisis guardados en Supabase
 */

import { supabase } from '../supabase-client.js';

export class AnalysisService {

    /**
     * Guardar un análisis en la base de datos
     * @param {Object} analysisData - Datos del análisis
     * @returns {Promise<Object>} Análisis guardado
     */
    static async save(analysisData) {
        try {
            const { data, error } = await supabase
                .from('analysis')
                .insert([{
                    harvester_id: analysisData.harvesterId || null,
                    sector_id: analysisData.sectorId || null,
                    batch_id: analysisData.batchId || null,
                    total_confirmed: analysisData.totalConfirmed,
                    ia_base: analysisData.iaBase,
                    manually_accepted: analysisData.manuallyAccepted || 0,
                    manually_added: analysisData.manuallyAdded || 0,
                    rejected: analysisData.rejected || 0,
                    doubts_reviewed: analysisData.doubtsReviewed || 0,
                    image_width: analysisData.imageWidth,
                    image_height: analysisData.imageHeight,
                    processing_time_ms: analysisData.processingTime,
                    notes: analysisData.notes || null
                }])
                .select()
                .single();

            if (error) throw error;

            console.log('✅ Análisis guardado:', data.id);
            return data;

        } catch (error) {
            console.error('❌ Error al guardar análisis:', error);
            throw error;
        }
    }

    /**
     * Obtener análisis recientes
     * @param {number} limit - Cantidad de resultados
     * @returns {Promise<Array>} Lista de análisis
     */
    static async getRecent(limit = 50) {
        try {
            const { data, error } = await supabase
                .from('v_analysis_detailed')
                .select('*')
                .limit(limit);

            if (error) throw error;
            return data;

        } catch (error) {
            console.error('❌ Error al obtener análisis:', error);
            throw error;
        }
    }

    /**
     * Obtener análisis por cosechador y rango de fechas
     * @param {string} harvesterId - ID del cosechador
     * @param {string} startDate - Fecha inicio (YYYY-MM-DD)
     * @param {string} endDate - Fecha fin (YYYY-MM-DD)
     * @returns {Promise<Array>} Lista de análisis
     */
    static async getByHarvester(harvesterId, startDate = null, endDate = null) {
        try {
            let query = supabase
                .from('analysis')
                .select('*')
                .eq('harvester_id', harvesterId);

            if (startDate) {
                query = query.gte('created_at', `${startDate}T00:00:00`);
            }
            if (endDate) {
                query = query.lte('created_at', `${endDate}T23:59:59`);
            }

            const { data, error } = await query.order('created_at', { ascending: false });

            if (error) throw error;
            return data;

        } catch (error) {
            console.error('❌ Error al obtener análisis por cosechador:', error);
            throw error;
        }
    }

    /**
     * Calcular total de estacas de un cosechador en una fecha específica
     * @param {string} harvesterId - ID del cosechador
     * @param {string} date - Fecha (YYYY-MM-DD)
     * @returns {Promise<Object>} Total de estacas
     */
    static async getTotalByDate(harvesterId, date) {
        try {
            const analyses = await this.getByHarvester(harvesterId, date, date);
            const total = analyses.reduce((sum, a) => sum + a.total_confirmed, 0);

            return {
                date,
                harvesterId,
                totalStakes: total,
                analysisCount: analyses.length,
                analyses
            };

        } catch (error) {
            console.error('❌ Error al calcular total:', error);
            throw error;
        }
    }
}
