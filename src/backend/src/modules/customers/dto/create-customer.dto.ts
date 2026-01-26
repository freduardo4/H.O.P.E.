import {
    IsString,
    IsOptional,
    IsEnum,
    IsEmail,
    IsObject,
    IsBoolean,
} from 'class-validator';
import { CustomerType } from '../entities/customer.entity';

export class CustomerPreferencesDto {
    @IsString()
    @IsOptional()
    contactMethod?: 'email' | 'phone' | 'sms';

    @IsBoolean()
    @IsOptional()
    receiveMarketing?: boolean;

    @IsString()
    @IsOptional()
    preferredLanguage?: string;
}

export class CreateCustomerDto {
    @IsEnum(CustomerType)
    @IsOptional()
    type?: CustomerType;

    @IsString()
    firstName: string;

    @IsString()
    lastName: string;

    @IsString()
    @IsOptional()
    companyName?: string;

    @IsEmail()
    email: string;

    @IsString()
    @IsOptional()
    phone?: string;

    @IsString()
    @IsOptional()
    alternatePhone?: string;

    @IsString()
    @IsOptional()
    address?: string;

    @IsString()
    @IsOptional()
    city?: string;

    @IsString()
    @IsOptional()
    state?: string;

    @IsString()
    @IsOptional()
    postalCode?: string;

    @IsString()
    @IsOptional()
    country?: string;

    @IsString()
    @IsOptional()
    taxId?: string;

    @IsString()
    @IsOptional()
    notes?: string;

    @IsObject()
    @IsOptional()
    preferences?: CustomerPreferencesDto;
}
