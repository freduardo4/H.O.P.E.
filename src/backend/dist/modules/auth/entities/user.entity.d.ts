export declare enum UserRole {
    ADMIN = "admin",
    SHOP_OWNER = "shop_owner",
    TECHNICIAN = "technician",
    VIEWER = "viewer"
}
export declare class User {
    id: string;
    email: string;
    passwordHash: string;
    firstName: string;
    lastName: string;
    role: UserRole;
    tenantId: string;
    isActive: boolean;
    refreshToken: string;
    lastLoginAt: Date;
    createdAt: Date;
    updatedAt: Date;
    get fullName(): string;
}
