import { IsString, IsNumber, Min, IsOptional } from 'class-validator';

export class ValidateFlashDto {
    @IsString()
    ecuId: string;

    @IsNumber()
    @Min(0)
    voltage: number;
}
