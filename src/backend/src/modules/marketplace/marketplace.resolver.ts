import { Resolver, Query, Mutation, Args, Context } from '@nestjs/graphql';
import { UseGuards } from '@nestjs/common';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { MarketplaceService } from './marketplace.service';
import { CalibrationListing } from './entities/calibration-listing.entity';
import { License } from './entities/license.entity';

@Resolver()
export class MarketplaceResolver {
    constructor(private readonly marketplaceService: MarketplaceService) { }

    @Query(() => [CalibrationListing])
    async calibrationListings(): Promise<CalibrationListing[]> {
        return this.marketplaceService.findAllListings();
    }

    @Query(() => CalibrationListing)
    async calibrationListing(@Args('id') id: string): Promise<CalibrationListing> {
        return this.marketplaceService.findListingById(id);
    }

    @Mutation(() => License)
    @UseGuards(JwtAuthGuard)
    async purchaseCalibration(
        @Args('listingId') listingId: string,
        @Args('hardwareId') hardwareId: string,
        @Context() context,
    ): Promise<License> {
        const user = context.req.user;
        return this.marketplaceService.createLicense(user, listingId, hardwareId);
    }

    @Query(() => Boolean)
    async verifyLicense(
        @Args('licenseKey') licenseKey: string,
        @Args('hardwareId') hardwareId: string,
    ): Promise<boolean> {
        return this.marketplaceService.validateLicense(hardwareId, licenseKey);
    }
}
