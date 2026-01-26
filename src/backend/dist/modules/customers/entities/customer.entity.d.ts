export declare enum CustomerType {
    INDIVIDUAL = "individual",
    BUSINESS = "business",
    FLEET = "fleet"
}
export declare class Customer {
    id: string;
    tenantId: string;
    type: CustomerType;
    firstName: string;
    lastName: string;
    companyName: string;
    email: string;
    phone: string;
    alternatePhone: string;
    address: string;
    city: string;
    state: string;
    postalCode: string;
    country: string;
    taxId: string;
    notes: string;
    preferences: {
        contactMethod?: 'email' | 'phone' | 'sms';
        receiveMarketing?: boolean;
        preferredLanguage?: string;
    };
    isActive: boolean;
    createdAt: Date;
    updatedAt: Date;
    get fullName(): string;
    get displayName(): string;
}
