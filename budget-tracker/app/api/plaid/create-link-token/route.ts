import { NextResponse } from 'next/server';
import { createLinkToken } from '@/lib/plaid';

export async function POST() {
  try {
    const linkToken = await createLinkToken('family-user');
    return NextResponse.json({ linkToken });
  } catch (error) {
    console.error('Error creating link token:', error);
    return NextResponse.json({ error: 'Failed to create link token' }, { status: 500 });
  }
}
