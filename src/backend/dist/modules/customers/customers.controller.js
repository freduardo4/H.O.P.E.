"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
var __param = (this && this.__param) || function (paramIndex, decorator) {
    return function (target, key) { decorator(target, key, paramIndex); }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.CustomersController = void 0;
const common_1 = require("@nestjs/common");
const customers_service_1 = require("./customers.service");
const dto_1 = require("./dto");
const customer_entity_1 = require("./entities/customer.entity");
const auth_1 = require("../auth");
const user_entity_1 = require("../auth/entities/user.entity");
let CustomersController = class CustomersController {
    constructor(customersService) {
        this.customersService = customersService;
    }
    async create(user, dto) {
        return this.customersService.create(user.tenantId, dto);
    }
    async findAll(user, type, search, city, page = 1, limit = 20) {
        return this.customersService.findAll({
            tenantId: user.tenantId,
            type,
            search,
            city,
            page: Number(page),
            limit: Number(limit),
        });
    }
    async getStats(user) {
        return this.customersService.getStats(user.tenantId);
    }
    async findOne(user, id) {
        return this.customersService.findOne(user.tenantId, id);
    }
    async findByEmail(user, email) {
        return this.customersService.findByEmail(user.tenantId, email);
    }
    async update(user, id, dto) {
        return this.customersService.update(user.tenantId, id, dto);
    }
    async remove(user, id) {
        await this.customersService.remove(user.tenantId, id);
        return { message: 'Customer deleted successfully' };
    }
};
exports.CustomersController = CustomersController;
__decorate([
    (0, common_1.Post)(),
    (0, auth_1.Roles)(user_entity_1.UserRole.ADMIN, user_entity_1.UserRole.SHOP_OWNER, user_entity_1.UserRole.TECHNICIAN),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User,
        dto_1.CreateCustomerDto]),
    __metadata("design:returntype", Promise)
], CustomersController.prototype, "create", null);
__decorate([
    (0, common_1.Get)(),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Query)('type')),
    __param(2, (0, common_1.Query)('search')),
    __param(3, (0, common_1.Query)('city')),
    __param(4, (0, common_1.Query)('page')),
    __param(5, (0, common_1.Query)('limit')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String, String, String, Object, Object]),
    __metadata("design:returntype", Promise)
], CustomersController.prototype, "findAll", null);
__decorate([
    (0, common_1.Get)('stats'),
    (0, auth_1.Roles)(user_entity_1.UserRole.ADMIN, user_entity_1.UserRole.SHOP_OWNER),
    __param(0, (0, auth_1.CurrentUser)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User]),
    __metadata("design:returntype", Promise)
], CustomersController.prototype, "getStats", null);
__decorate([
    (0, common_1.Get)(':id'),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Param)('id', common_1.ParseUUIDPipe)),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String]),
    __metadata("design:returntype", Promise)
], CustomersController.prototype, "findOne", null);
__decorate([
    (0, common_1.Get)('email/:email'),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Param)('email')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String]),
    __metadata("design:returntype", Promise)
], CustomersController.prototype, "findByEmail", null);
__decorate([
    (0, common_1.Patch)(':id'),
    (0, auth_1.Roles)(user_entity_1.UserRole.ADMIN, user_entity_1.UserRole.SHOP_OWNER, user_entity_1.UserRole.TECHNICIAN),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Param)('id', common_1.ParseUUIDPipe)),
    __param(2, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String, dto_1.UpdateCustomerDto]),
    __metadata("design:returntype", Promise)
], CustomersController.prototype, "update", null);
__decorate([
    (0, common_1.Delete)(':id'),
    (0, auth_1.Roles)(user_entity_1.UserRole.ADMIN, user_entity_1.UserRole.SHOP_OWNER),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Param)('id', common_1.ParseUUIDPipe)),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String]),
    __metadata("design:returntype", Promise)
], CustomersController.prototype, "remove", null);
exports.CustomersController = CustomersController = __decorate([
    (0, common_1.Controller)('customers'),
    (0, common_1.UseGuards)(auth_1.JwtAuthGuard, auth_1.RolesGuard),
    __metadata("design:paramtypes", [customers_service_1.CustomersService])
], CustomersController);
//# sourceMappingURL=customers.controller.js.map