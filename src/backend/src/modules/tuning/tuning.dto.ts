
export class OptimizeRequestDto {
    current_map: number[][];
    target_map: number[][];
    datalog_mask?: number[][];
    config?: {
        generations?: number;
        population_size?: number;
        mutation_rate?: number;
        crossover_rate?: number;
        mutation_strength?: number;
    };
}

export class OptimizeResponseDto {
    optimized_map: number[][];
    fitness_history: number[];
    status: string;
}
