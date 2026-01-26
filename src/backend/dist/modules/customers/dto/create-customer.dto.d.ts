import { CustomerType } from '../entities/customer.entity';
export declare class CustomerPreferencesDto {
    contactMethod?: 'email' | 'phone' | 'sms';
    receiveMarketing?: boolean;
    preferredLanguage?: string;
}
export declare class CreateCustomerDto {
    type?: CustomerType;
    firstName: string;
    lastName: string;
    companyName?: string;
    email: string;
    phone?: string;
    alternatePhone?: string;
    address?: string;
    city?: string;
    state?: string;
    postalCode?: string;
    country?: string;
    taxId?: string;
    notes?: string;
    preferences?: CustomerPreferencesDto;
}
