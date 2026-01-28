import { IsString, IsArray, IsOptional, IsNotEmpty } from 'class-validator';

export class CreateWikiPostDto {
    @IsString()
    @IsNotEmpty()
    title: string;

    @IsString()
    @IsNotEmpty()
    content: string;

    @IsArray()
    @IsString({ each: true })
    @IsOptional()
    tags?: string[];
}

export class SearchWikiFixDto {
    @IsString()
    @IsOptional()
    query?: string;

    @IsString()
    @IsOptional()
    dtc?: string;

    @IsOptional()
    page?: number;

    @IsOptional()
    limit?: number;
}
