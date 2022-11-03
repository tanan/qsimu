def sql_for_find_job() -> str:
    return """
        SELECT
            id,
            creation_time,
            execution_second,
            nqubit,
            depth,
            gate_type,
            gate_set,
            bn_type,
            bn_range,
            bn,
            cn,
            r,
            t_type,
            max_time,
            min_time,
            t,
            cost,
            parameter,
            iteration,
            cost_history,
            parameter_history,
            iteration_history,
            noise_singlequbit_enabled,
            noise_singlequbit_value,
            noise_twoqubit_enabled,
            noise_twoqubit_value,
            config
        FROM jobs
    """


def sql_for_find_job_by_time() -> str:
    return """
        SELECT
            id,
            creation_time,
            execution_second,
            nqubit,
            depth,
            gate_type,
            gate_set,
            bn_type,
            bn_range,
            bn,
            cn,
            r,
            t_type,
            max_time,
            min_time,
            t,
            cost,
            parameter,
            iteration,
            cost_history,
            parameter_history,
            iteration_history
        FROM jobs
        WHERE
            nqubit = ?
        AND
            depth = ?
        AND
            gate_type = ?
        AND
            t_type = ?
        AND
            min_time = ?
        AND
            max_time = ?
    """
