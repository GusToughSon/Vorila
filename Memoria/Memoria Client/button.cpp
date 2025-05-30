#include "main.h"

void button::pressed()
{
	switch (action)
	{

	case 1: // Secure trade: exit button
		st->tradeabort();
		st->clear();
		break;
	case 2: // Secure trade: scroll up
		st->scroll -= 3;
		if (st->scroll < 0) st->scroll = 0;
		break;	
	case 3: // Secure trade: scroll down
		st->scroll += 3;
		if (st->scroll > (MaxBackpackItems - 12)) st->scroll = (MaxBackpackItems - 12);
		break;	
	case 4: // Secure trade: lock
		st->accept();
		break;


	case 10: // Deed: scroll up

		Clan->Scroll -= 3;
		if (Clan->Scroll < 0) Clan->Scroll = 0;
		break;

	case 11: // Deed: scroll down

		Clan->Scroll += 3;
		if (Clan->Scroll + 32 > 50) Clan->Scroll = 50 - 32;
		break;

	case 12: // GMtool: scroll up

		GMTool->Scroll -= 3;

		if (GMTool->Scroll < 0) 
			GMTool->Scroll = 0;

//		GMTool->GetListView();

		break;

	case 13: // GMTool: scroll down

		GMTool->Scroll += 3;

		if (GMTool->Scroll > m_Connections - 33) 
			GMTool->Scroll = m_Connections - 33;

//		GMTool->GetListView();

		break;

	case 14: // GMTool: exit button

		GMTool->On = false;

		break;

	case 15: // GMTool: All button

		buffer[0] = 60;
		buffer[1] = 4;
		send(Socket, buffer, 2, 0);

		buffer[0] = 60;
		buffer[1] = 49;
		send(Socket, buffer, 2, 0);

		if (!Client.SayOn)
			Client.SayOn = true;
		if (!Client.WhisOn)
			Client.WhisOn = true;
		if (!Client.GlobOn)
			Client.GlobOn = true;

		buffer[0] = 60;
		buffer[1] = 5;
		send(Socket, buffer, 2, 0);

		break;

	case 16: // BS scroll up

		BS->scroll -= 3;
		if (BS->scroll < 0) BS->scroll = 0;
		break;

	case 17: // BS scroll down

		BS->scroll += 3;
		if (BS->scroll > 50) BS->scroll = 50;
		break;

	case 18: // Close Backpackview and Bankview
		
		GMTool->bBackpackView = false;
		GMTool->bBankView = false;

		break;

	case 100:

		GMTool->Changemode = true;
		GMTool->Changetarget = 0;

		break;

	case 101:

		GMTool->Changemode = true;
		GMTool->Changetarget = 1;

		break;

	case 102:

		GMTool->Changemode = true;
		GMTool->Changetarget = 2;

		break;

	case 103:

		GMTool->AM[0]--;
		if (GMTool->AM[0] < 0) GMTool->AM[0] = 0;

		break;

	case 104:

		GMTool->AM[1]--;
		if (GMTool->AM[1] < 0) GMTool->AM[1] = 0;

		break;

	case 105:

		GMTool->AM[2]--;
		if (GMTool->AM[2] < 0) GMTool->AM[2] = 0;

		break;

	case 106:

		GMTool->AM[0]++;
		if (GMTool->AM[0] > 10) GMTool->AM[0] = 10;

		break;

	case 107:

		GMTool->AM[1]++;
		if (GMTool->AM[1] > 10) GMTool->AM[1] = 10;

		break;

	case 108:

		GMTool->AM[2]++;
		if (GMTool->AM[2] > 10) GMTool->AM[2] = 10;

		break;

	case 109:
		GMTool->itemType--;
		if (GMTool->itemType < 0) GMTool->itemType = 0;

		break;

	case 110:

		GMTool->itemID--;
		if (GMTool->itemID < 0) GMTool->itemID = 0;

		break;

	case 111:

		GMTool->itemAm--;
		if (GMTool->itemAm < 0) GMTool->itemAm = 0;

		break;

	case 112:

		GMTool->itemType++;
		if (GMTool->itemType > 20) GMTool->itemType = 20;

		break;

	case 113:

		GMTool->itemID++;
		if (GMTool->itemID > 200) GMTool->itemID = 200;

		break;

	case 114:

		GMTool->itemAm++;
//		if (GMTool->itemAm > 20000) GMTool->itemAm = 20000; -- No amount limit

		break;

	}
}



button::button(class buttoncontrol *base2, short x2, short y2, short h, short w, short pic2, short action2)
{
	base = base2;
	x = x2;
	y = y2;
	height = h;
	width = w;
	pic = pic2;
	action = action2;
}

button::~button()
{

}


bool button::ison(short x2, short y2)
{
	if (((x - x2) * -1) <= width && ((x - x2) * -1) >= 0 &&
		((y - y2) * -1) <= height && ((y - y2) * -1) >= 0)
		return true;
	else return false;
}


void button::draw()
{
	short pic2;
	if (ison(MX, MY) && Client.mbut1)
		pic2 = pic + 1;
	else pic2 = pic;

	SDL_Rect dst;
	dst.x = x;
	dst.y = y;
	dst.h = sdl.Rect_addgf[pic2].h;
	dst.w = sdl.Rect_addgf[pic2].w;

	SDL_BlitSurface(G.addgf, &sdl.Rect_addgf[pic2], Screen.screen, &dst);	
}

void button::addto()
{
	if (base->buttons == NULL)
	{
		next = NULL;
		prev = NULL;
		base->buttons = this;
	}
	else
	{
		next = base->buttons;
		prev = NULL;
		base->buttons->prev = this;
		base->buttons = this;
	}	
}

void button::remove()
{
	if (base->buttons == this)
	{
		base->buttons = base->buttons->next;
		base->buttons->prev = NULL;
	}
	else
	{
		if (prev != NULL)
			prev->next = next;
		if (next != NULL)
			next->prev = prev;			
	}
	delete this;
}

void buttoncontrol::clear()
{
	button *tmp;
	while (buttons)
	{
		tmp = buttons->next;
		delete buttons;
		buttons = tmp;	
	}
}

void buttoncontrol::mouseclick()
{
	button *but;
	but = buttons;
	while (but)
	{
		if (but->ison(MX, MY))
		{
			but->pressed();
			
			but = NULL;
		}
		if (but) but = but->next;
	}
}

void buttoncontrol::draw()
{
	button *tmp;
	tmp = buttons;
	while (tmp)
	{
		tmp->draw();
		tmp = tmp->next;
	}
}

buttoncontrol::buttoncontrol()
{
	buttons = NULL;
}

buttoncontrol::~buttoncontrol()
{
	clear();
}