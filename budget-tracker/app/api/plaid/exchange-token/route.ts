import { NextResponse } from 'next/server';
import { exchangePublicToken, getAccounts } from '@/lib/plaid';
import { prisma } from '@/lib/db';

export async function POST(request: Request) {
  try {
    const { publicToken, institutionName } = await request.json();

    const { accessToken, itemId } = await exchangePublicToken(publicToken);

    // Save the Plaid item
    await prisma.plaidItem.create({
      data: {
        itemId,
        accessToken,
      },
    });

    // Get and save accounts
    const accounts = await getAccounts(accessToken);

    for (const account of accounts) {
      await prisma.bankAccount.create({
        data: {
          plaidAccountId: account.account_id,
          plaidItemId: itemId,
          accessToken,
          institutionName,
          accountName: account.name,
          accountType: account.type,
        },
      });
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error exchanging token:', error);
    return NextResponse.json({ error: 'Failed to connect bank' }, { status: 500 });
  }
}
